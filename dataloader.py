import os
import pydicom as dicom
import glob
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from scipy.spatial.transform import Rotation as R
import pymesh


def transform(img):
    max_value = img.max()
    img_tensor = torch.from_numpy(img.astype(np.int32))
    img_tensor = img_tensor.float() / max_value
    return img_tensor

class Data(Dataset):
    def __init__(self, root, transform):
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
        CT = os.path.join(self.CT[index])
        Xray = os.path.join(self.Xray[index])
        angle = self.rotation[index]

        # r = R.from_euler('z', angle, degrees=True)
        # r = r.as_matrix()
        label = np.array([0, 0, angle, 0, 0, 0], dtype=np.float32)
        CT_list = os.listdir(CT)

        CT_out = []
        for file in sorted(CT_list):
            CT_out.append(np.array(Image.open(os.path.join(CT, file))))

        CT_out = np.expand_dims(np.array(CT_out, dtype=np.float32), axis=-1).transpose((3, 1, 2, 0))
        ct_min = np.min(np.min(np.min(CT_out, axis=0), axis=0), axis=0)
        ct_max = np.max(np.max(np.max(CT_out, axis=0), axis=0), axis=0)
        CT_out = (CT_out - ct_min) / ct_max

        Xray_out = np.expand_dims(np.array(Image.open(Xray), dtype=np.float32), axis=-1).transpose((2, 0, 1))
        xray_min = np.min(np.min(np.min(Xray_out, axis=0), axis=0), axis=0)
        xray_max = np.max(np.max(np.max(Xray_out, axis=0), axis=0), axis=0)
        Xray_out = (Xray_out - xray_min) / xray_max

        return torch.tensor(CT_out), torch.tensor(Xray_out), label, self.num[index]
    def __len__(self):
        return self.num_samples



class Kaist_Data(Data):
    def __init__(self, root, transform):
        self.CT = pymesh.load_mehs('hepatic artery_200417.stl')
        self.dlist = os.listdir(root)

    def __getitem__(self, item):
        self.xray = self.dlist[item]
        return self.CT, self.xray

    def __len__(self):
        return len(self.dlist)



if __name__ == "__main__":
    root = './registration/2D3D_Data/'
    path = '/home/srk1995/pub/db/kaist_vessel/'
    # cTdataloader = Data(root, transform=transforms.ToTensor())
    kdata = Kaist_Data(path, transform=transforms.ToTensor())
    for i, data in enumerate(kdata):
        print(data)

    print("EOP")