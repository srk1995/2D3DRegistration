import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "9,8"

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from utils import cartesian_product
import utils
import visdom


def transform(img):
    max_value = img.max()
    img_tensor = torch.from_numpy(img.astype(np.int32))
    img_tensor = img_tensor.float() / max_value
    return img_tensor



class SegData(Dataset):
    def __init__(self, root, train, transform):
        """
        :param root: the path of data
        :param transform: transforms to make the output tensor

        """
        self.root = root
        self.s = 1
        self.dlist = [os.path.join(self.root, x) for x in os.listdir(root)]
        self.transform = transform
        self.zeros = np.array([0], dtype=np.float64).reshape(-1)
        self.rotation = np.array([np.linspace(-30, 30, 15)] * self.s).reshape(-1)
        self.translation = np.array([np.linspace(-5, 5, 3)] * self.s).reshape(-1)
        self.label = cartesian_product(self.zeros, self.zeros, self.rotation, self.zeros, self.zeros, self.translation)
        self.CT = []

        # self.drr_win = None
        # self.vis = visdom.Visdom()

        # self.num_samples = len(self.dlist)
        if train:
            file = open('train_zz.csv', 'w')
        else:
            file = open('test_zz.csv', 'w')
        for f in self.dlist:
            # path = os.path.join(f, 'xray_256_complex')
            # if not os.path.isdir(path):
            #     os.mkdir(path)
            CT = os.path.join(f, 'numpy_RG_npy.npy')

            # CT_out = np.load(CT)
            # CT_out = np.expand_dims(np.array(CT_out, dtype=np.float32), axis=-1).transpose((3, 2, 1, 0))
            # CT_out = torch.tensor(CT_out)
            # T = torch.zeros(6, dtype=torch.float32)
            for i, T in enumerate(self.label, 1):
                # drr = utils.DRR_generation(torch.tensor(CT_out), torch.tensor(T, dtype=torch.float32).view(1, 6), 1)
                # drr_path = os.path.join(path, str(int(i)))
                # np.save(drr_path, drr.cpu().numpy())
                m = "{},{}_{}_{}_{}_{}_{}\n".format(CT, str(T[0]), str(T[1]), str(T[2]), str(T[3]), str(T[4]), str(T[5]))
                file.write(m)

        # im = drr.view((960, 1240)).cpu().numpy()
        # self.drr_win = utils.PlotImage(vis=self.vis, img=im, win=self.drr_win, title="DRR")

        # ct_mean = torch.mean(CT_out)
        # ct_std = torch.std(CT_out)
        # CT_out = (CT_out - ct_mean) / ct_std
        file.close()

    def __getitem__(self, index):
        """

        :param index:
        :return: CT_out: [C, D, H, W] == [1, 393, 512, 512]
        :return drr: [C, H, W]
        :return T : [6]
        """
        return 0

    def __len__(self):
        return self.label.size


if __name__ == "__main__":
    # train_path = './registration/2D3D_Data/train'
    # test_path = './registration/2D3D_Data/test'
    train_path = '/home/srk1995/pub/db/Unet_1024/Train/'
    test_path = '/home/srk1995/pub/db/Unet_1024/Test/'

    # cTdataloader = Data(root, transform=transforms.ToTensor())
    kdata_train = SegData(train_path, train=True, transform=transforms.ToTensor())
    kdata_test = SegData(test_path, train=False, transform=transforms.ToTensor())

    # for i, data in enumerate(trainloader):
    #     print(data)

    print("EOP")