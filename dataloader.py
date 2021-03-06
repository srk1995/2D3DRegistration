import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import pymesh
import visdom
import utils
import skeleton2
from scipy import ndimage
import argparse
import time




def transform(img):
    max_value = img.max()
    img_tensor = torch.from_numpy(img.astype(np.int32))
    img_tensor = img_tensor.float() / max_value
    return img_tensor

class Data(Dataset):
    def __init__(self, root, transform):
        """
        :param root: the path of data
        :param transform: transforms to make the output tensor
        
        """
        self.root = root
        self.dlist = os.listdir(root)
        self.transform = transform
        self.CT = []
        self.Xray = []
        self.num = []
        if self.root[-5:] == 'train':
            self.rotation = np.loadtxt('rotation.txt')
        else:
            self.rotation = np.loadtxt('rotation_test.txt')
        num_samples = 0

        for i in self.dlist:
            set = os.path.join(self.root, i)
            set_inner = os.path.join(set, os.listdir(set)[0])
            lists = sorted(os.listdir(set_inner))
            cnt = sum('CINE' in s for s in lists)

            CT = ['ANGIO' in s for s in lists]
            CT = [j for j, x in enumerate(CT) if x]
            Xray = ['CINE' in s for s in lists]
            Xray = [j for j, x in enumerate(Xray) if x]


            # CT = np.tile(CT, (1, len(Xray)))[0]
            for j in range(cnt):
                t_list = sorted(os.listdir(os.path.join(set_inner, lists[Xray[j]], 'image')))
                t_cnt = sum('png' in s for s in t_list)
                CT = np.tile(CT, (1, t_cnt))[0]
                for k in range(t_cnt):
                    self.num.append(i[-1])
                    self.CT.append(os.path.join(set_inner, lists[CT[k]], 'image'))
                    self.Xray.append(os.path.join(set_inner, lists[Xray[j]], 'image', t_list[k]))

                num_samples += t_cnt

        self.num_samples = num_samples


    def __getitem__(self, index):
        """
        :param index:
        :return CT: [B, C, H, W, D] == [4, 1, 512, 512, 393]
        :return X-ray: [B, C, H, W] == [4, 1, 960, 1240]
        """
        CT = os.path.join(self.CT[index])
        Xray = os.path.join(self.Xray[index])
        angle = self.rotation[index]

        # r = R.from_euler('z', angle, degrees=True)
        # r = r.as_matrix()
        label = torch.tensor([0, 0, angle, 0, 0, 0], dtype=torch.float32)
        CT_list = os.listdir(CT)

        CT_out = []
        for file in sorted(CT_list):
            CT_out.append(np.array(Image.open(os.path.join(CT, file))))

        CT_out = torch.tensor(CT_out, dtype=torch.float32)
        CT_out = CT_out.view((1, CT_out.shape[0], CT_out.shape[1], CT_out.shape[2]))
        ct_mean = torch.mean(CT_out)
        ct_std = torch.std(CT_out)
        CT_out = (CT_out - ct_mean) / ct_std

        Xray_out = torch.tensor(np.expand_dims(np.array(Image.open(Xray), dtype=np.float32), axis=-1).transpose((2, 0, 1)), dtype=torch.float32)
        Xray_mean = torch.mean(Xray_out)
        Xray_std = torch.std(Xray_out)
        Xray_out = (Xray_out - Xray_mean) / Xray_std

        return CT_out, Xray_out, label

    def __len__(self):
        return self.num_samples


class SegData(Dataset):
    def __init__(self, root, transform):
        """
        :param root: the path of data
        :param transform: transforms to make the output tensor

        """
        self.root = root
        self.dlist = [os.path.join(self.root, x) for x in os.listdir(root)]
        self.transform = transform

        self.xray_list = []
        for f in self.dlist:
            self.xray_list.append([os.path.join(f, 'xray_256_x', x) for x in os.listdir(os.path.join(f, 'xray_256_x'))])

        self.xray_list = np.array(self.xray_list).reshape(-1)
        self.drr_win = None
        self.vis = visdom.Visdom()

        # self.num_samples = len(self.dlist)

    def __getitem__(self, index):
        """

        :param index:
        :return: CT_out: [C, D, H, W] == [1, 393, 512, 512]
        :return drr: [C, H, W]
        :return T : [6]
        """

        tt = self.xray_list[index].split("/")
        path = ''
        for t in tt[:-2]:
            path += t + '/'
        CT = os.path.join(path, '3d_numpy.npy')

        CT_out = np.load(CT)
        CT_out = np.expand_dims(np.array(CT_out, dtype=np.float32), axis=-1).transpose((3, 2, 1, 0))
        CT_out = torch.tensor(CT_out)

        xray = np.load(self.xray_list[index])
        rot = tt[-1].split(".")[0]

        if rot[0] == '_':
            r = -float(rot[1:])
        else:
            r = float(rot)
        T = torch.zeros(6, dtype=torch.float32)
        T[2] = torch.tensor(r)


        ct_mean = torch.mean(CT_out)
        ct_std = torch.std(CT_out)
        CT_out = (CT_out - ct_mean) / ct_std

        return CT_out, torch.tensor(xray, dtype=torch.float32), T

    def __len__(self):
        return len(self.xray_list)

class SegData_csv(Dataset):
    def __init__(self, file, proj_pix, transform):
        """
        :param root: the path of data
        :param transform: transforms to make the output tensor

        """
        self.dlist = np.loadtxt(file, delimiter=",", dtype=str)
        self.transform = transform
        self.proj_pix = proj_pix

        self.drr_win = None
        self.vis = visdom.Visdom()

        # self.num_samples = len(self.dlist)

    def __getitem__(self, index):
        """

        :param index:
        :return: CT_out: [C, D, H, W] == [1, 393, 512, 512]
        :return drr: [C, H, W]
        :return T : [6]
        """

        tt = self.dlist[index, :]

        CT = os.path.join(tt[0])

        CT= np.load(CT)
        CT_out = np.expand_dims(np.array(CT, dtype=np.float32), axis=-1).transpose((3, 2, 1, 0))
        CT_out = torch.tensor(CT_out)

        T = torch.tensor(np.array(tt[1:], dtype=np.float32), dtype=torch.float32)

        # tic = time.clock()
        catheter = []
        while len(catheter) == 0:
            a = skeleton2.mapping(CT)
            skel = a.skel

            xyz = np.where(skel == 1)
            idx = np.random.randint(len(xyz[0]), size=2)
            sp = np.array([xyz[0][idx[0]], xyz[1][idx[0]], xyz[2][idx[0]]])
            fp = np.array([xyz[0][idx[1]], xyz[1][idx[1]], xyz[2][idx[1]]])
            catheter = a.get_road(sp, fp)

        catheter = np.array(catheter)
        C = np.zeros_like(CT)
        C[catheter[:, 0], catheter[:, 1], catheter[:, 2]] = 1
        C = np.expand_dims(np.array(C, dtype=np.float32), axis=-1).transpose((3, 2, 1, 0))
        C = torch.tensor(C)

        xray = utils.DRR_generation(torch.tensor(C), T.view(1, 6), 1, self.proj_pix)
        # toc = time.clock()
        # print(toc - tic)

        if (CT_out.min() != 0) or (CT_out.max() != 1):
            print(CT)
        ct_mean = torch.mean(CT_out)
        ct_std = torch.std(CT_out)
        CT_out = (CT_out - ct_mean) / ct_std

        return CT_out, xray, T

    def __len__(self):
        return len(self.dlist)


class SegData_catheter(Dataset):
    def __init__(self, file, proj_pix, transform):
        """
        :param root: the path of data
        :param transform: transforms to make the output tensor

        """
        self.dlist = np.loadtxt(file, delimiter=",", dtype=str)
        self.transform = transform
        self.proj_pix = proj_pix

        self.drr_win = None
        self.vis = visdom.Visdom()
        self.tt = 0
        self.ttt = 0

        # self.num_samples = len(self.dlist)

    def __getitem__(self, index):
        """

        :param index:
        :return: CT_out: [C, D, H, W] == [1, 393, 512, 512]
        :return drr: [C, H, W]
        :return T : [6]
        """

        tt = self.dlist[index, :]

        CT = os.path.join(tt[0])

        CT= np.load(CT)
        CT_out = np.expand_dims(np.array(CT, dtype=np.float32), axis=-1).transpose((3, 2, 1, 0))
        CT_out = torch.tensor(CT_out)

        T = torch.tensor(np.array(tt[1:][0].split('_'), dtype=np.float32), dtype=torch.float32)

        # tic = time.clock()
        # catheter = []
        # while len(catheter) == 0:
        #     a = skeleton2.mapping(CT)
        #     skel = a.skel
        #
        #     xyz = np.where(skel == 1)
        #     idx = np.random.randint(len(xyz[0]), size=2)
        #     sp = np.array([xyz[0][idx[0]], xyz[1][idx[0]], xyz[2][idx[0]]])
        #     fp = np.array([xyz[0][idx[1]], xyz[1][idx[1]], xyz[2][idx[1]]])
        #     catheter = a.get_road(sp, fp)
        #
        # catheter = np.array(catheter)
        # C = np.zeros_like(CT)
        # C[catheter[:, 0], catheter[:, 1], catheter[:, 2]] = 1
        # C = np.expand_dims(np.array(C, dtype=np.float32), axis=-1).transpose((3, 2, 1, 0))
        # C = torch.tensor(C)

        # xray = utils.DRR_generation(torch.tensor(C), T.view(1, 6), 1, self.proj_pix)
        # toc = time.clock()
        # print(toc - tic)

        ttt = tt[0].split('/')
        ttt = ttt[:-1]
        t = '/'
        for x in ttt:
            t = os.path.join(t, x)
        xray = np.load(os.path.join(t, tt[-1] + '.npy'))

        if (CT_out.min() != 0) or (CT_out.max() != 1):
            print(CT)
        self.tt += sum(sum(sum(CT == 1)))
        self.ttt += 1
        ct_mean = torch.mean(CT_out)
        ct_std = torch.std(CT_out)
        CT_out = (CT_out - ct_mean) / ct_std

        return CT_out, xray, T, self.tt / self.ttt

    def __len__(self):
        return len(self.dlist)




class SegData_catheter_pt(Dataset):
    def __init__(self, file, proj_pix, c, transform):
        """
        :param root: the path of data
        :param transform: transforms to make the output tensor

        """
        self.dlist = np.loadtxt(file, delimiter=",", dtype=str)
        self.transform = transform
        self.proj_pix = proj_pix

        self.drr_win = None
        self.vis = visdom.Visdom()
        self.c = c
        self.num_v = np.inf
        self.num_c = np.inf

        # self.num_samples = len(self.dlist)

    def __getitem__(self, index):
        """

        :param index:
        :return: CT_out: [C, D, H, W] == [1, 393, 512, 512]
        :return drr: [C, H, W]
        :return T : [6]
        """
        while True:
            tt = self.dlist[index, :]
            val = [-30, 30, -5, 5]

            CT = os.path.join(tt[0])

            CT= np.load(CT)
            CT_v = np.expand_dims(np.array(CT, dtype=np.float32), axis=-1).transpose((3, 2, 1, 0))
            CT_v = torch.tensor(CT_v)
            str_3D = ndimage.morphology.generate_binary_structure(3, 1)
            big_str_3D = ndimage.morphology.iterate_structure(str_3D, 3)
            arr_out_3D = ndimage.morphology.binary_erosion(CT_v[0], big_str_3D)

            T = torch.tensor(np.array(tt[1:][0].split('_'), dtype=np.float32), dtype=torch.float32)
            T = utils.OnehotEncoding(T[2], val, self.c)

            ttt = tt[0].split('/')
            ttt = ttt[:-1]
            t = '/'
            for x in ttt:
                t = os.path.join(t, x)
            xray_v = np.load(os.path.join(t, tt[-1] + '.npy'))

            CT_out = np.array(np.where(F.interpolate(torch.tensor(arr_out_3D, dtype=torch.float32), size=128)))
            # CT_out = np.array(np.where(F.interpolate(CT_v[0], size=128)))
            xray = np.array(np.where(xray_v[0] != xray_v.min()))

            if (CT_out.shape[1] > 5000) & (xray.shape[1] > 400):
                CT_out = CT_out[:, np.random.randint(CT_out.shape[1], size=5000)]
                xray = xray[:, np.random.randint(xray.shape[1], size=400)]
                break
            else:
                index = np.random.randint(len(self.dlist))


        return CT_out, xray, T

    def __len__(self):
        return len(self.dlist)

class SegData_catheter_pt_tr(Dataset):
    def __init__(self, file, proj_pix, c, transform):
        """
        :param root: the path of data
        :param transform: transforms to make the output tensor

        """
        self.dlist = np.loadtxt(file, delimiter=",", dtype=str)
        self.transform = transform
        self.proj_pix = proj_pix

        self.drr_win = None
        self.vis = visdom.Visdom()
        self.c = c

        # self.num_samples = len(self.dlist)

    def __getitem__(self, index):
        """

        :param index:
        :return: CT_out: [C, D, H, W] == [1, 393, 512, 512]
        :return drr: [C, H, W]
        :return T : [6]
        """

        tt = self.dlist[index, :]

        CT = os.path.join(tt[0])

        CT= np.load(CT)
        CT_v = np.expand_dims(np.array(CT, dtype=np.float32), axis=-1).transpose((3, 2, 1, 0))
        CT_v = torch.tensor(CT_v)
        str_3D = ndimage.morphology.generate_binary_structure(3, 1)
        big_str_3D = ndimage.morphology.iterate_structure(str_3D, 3)
        arr_out_3D = ndimage.morphology.binary_erosion(CT_v[0], big_str_3D)

        T = torch.tensor(np.array(tt[1:][0].split('_'), dtype=np.float32), dtype=torch.float32)
        T = utils.OnehotEncoding(T, self.c)

        # tic = time.clock()
        # catheter = []
        # while len(catheter) == 0:
        #     a = skeleton2.mapping(CT)
        #     skel = a.skel
        #
        #     xyz = np.where(skel == 1)
        #     idx = np.random.randint(len(xyz[0]), size=2)
        #     sp = np.array([xyz[0][idx[0]], xyz[1][idx[0]], xyz[2][idx[0]]])
        #     fp = np.array([xyz[0][idx[1]], xyz[1][idx[1]], xyz[2][idx[1]]])
        #     catheter = a.get_road(sp, fp)
        #
        # catheter = np.array(catheter)
        # C = np.zeros_like(CT)
        # C[catheter[:, 0], catheter[:, 1], catheter[:, 2]] = 1
        # C = np.expand_dims(np.array(C, dtype=np.float32), axis=-1).transpose((3, 2, 1, 0))
        # C = torch.tensor(C)

        # xray = utils.DRR_generation(torch.tensor(C), T.view(1, 6), 1, self.proj_pix)
        # toc = time.clock()
        # print(toc - tic)

        ttt = tt[0].split('/')
        ttt = ttt[:-1]
        t = '/'
        for x in ttt:
            t = os.path.join(t, x)
        xray_v = np.load(os.path.join(t, tt[-1] + '.npy'))

        CT_out = np.array(np.where(F.interpolate(torch.tensor(arr_out_3D, dtype=torch.float32), size=128)))
        # CT_out = np.array(np.where(F.interpolate(CT_v[0], size=128)))
        xray = np.array(np.where(xray_v[0] != xray_v.min()))

        return CT_out, xray, T, CT_v, xray_v, self.c

    def __len__(self):
        return len(self.dlist)


class Kaist_Data(Dataset):
    def __init__(self, root, transform):
        self.root = root
        self.transform = transform
        self.CT = pymesh.load_mesh(self.root + 'hepatic_artery_200417.stl')
        self.dlist = os.listdir(self.root)
        self.list = [root + x for x in self.dlist if os.path.isdir(root+x)]
        self.list.sort()
        self.xlist = []
        self.num = 0
        for x in self.list:
            tt = os.listdir(x)
            tt.sort()
            [self.xlist.append(x + '/' + p) for p in tt]
            self.num += len(tt)

        self.f = open("kaist_rt.txt", "r")
        self.lines = self.f.read().split('\n')
        self.rt = np.zeros((len(self.lines), 4))
        for i in range(len(self.lines)):
            # print(self.lines[i].split(' '))
            self.rt[i, :] = np.asarray(self.lines[i].split(' '))

    def __getitem__(self, item):
        self.xray_path = self.xlist[item]
        self.rot = self.xray_path.split('/')[-2]
        if self.rot[0] == '_':
            self.rot = '-' + self.rot[1:]

        self.rot = int(self.rot)
        self.label = self.rt[np.where(self.rt[:, 0] == self.rot)[0][0], :]
        self.label = np.concatenate(([0, 0], self.label), axis=0)
        self.xray = np.array(Image.open(self.xray_path))

        return self.transform(self.CT), self.transform(self.xray)

    def __len__(self):
        return self.num


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preocess some numbers.")
    parser.add_argument('--net', type=str, help='Network architecture, 6layer, 8layer, unet, homo, homo_bn', default='pointnet2')
    parser.add_argument('--alpha', type=float, help='alpha', default=1e-4)
    parser.add_argument('--lr', type=float, help='Learning rate', default=1e-3)
    parser.add_argument('--gpu', type=str, help='gpu number', default='9')
    parser.add_argument('--qt', type=int, help='The number of bins', default=1024)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    env = "seg_" + args.net + "_alpha_" + str(args.alpha) + "_lr_" + str(args.lr) + "_qt_" + str(args.qt)

    train_file = './train_zz.csv'
    test_file = './test_zz.csv'
    PATH = './saved/'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vis = visdom.Visdom()

    train_batch_num = 1
    alpha = args.alpha

    proj_pix = [256, 256]
    val = [-30, 30, -5, 5]
    lb = utils.OnehotDecoding(np.repeat(np.array([i for i in range(args.qt)]), 6).reshape(-1, 6), val, args.qt)

    train_file = './train_z.csv'
    test_file = './test_z.csv'

    transfroms_ = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Resize((64, 64))
    ])

    train_dataset = SegData_catheter_pt(train_file, proj_pix, args.qt, transform=transfroms_)
    test_dataset = SegData_catheter_pt(test_file, proj_pix, args.qt, transform=transfroms_)
    trainloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
    testloader = DataLoader(test_dataset, batch_size=1, shuffle=False,num_workers=0)
    for i, data in enumerate(trainloader):
        print("")

    for i, data in enumerate(testloader):
        print("")

    print("EOP")