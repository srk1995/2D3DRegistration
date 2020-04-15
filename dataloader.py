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
        self.rotation = np.loadtxt('rotation.txt')
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
        Xray_out = np.expand_dims(np.array(Image.open(Xray), dtype=np.float32), axis=-1).transpose((2, 0, 1))

        return torch.tensor(CT_out), torch.tensor(Xray_out), label, self.num[index]
    def __len__(self):
        return self.num_samples



if __name__ == "__main__":
    root = './registration/2D3D_Data/'
    cTdataloader = Data(root, transform=transforms.ToTensor())
    for i, data in enumerate(cTdataloader):
        print(data)

    print("EOP")