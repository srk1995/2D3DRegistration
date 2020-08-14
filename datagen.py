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
import utils
import visdom


def transform(img):
    max_value = img.max()
    img_tensor = torch.from_numpy(img.astype(np.int32))
    img_tensor = img_tensor.float() / max_value
    return img_tensor



class SegData(Dataset):
    def __init__(self, root, transform):
        """
        :param root: the path of data
        :param transform: transforms to make the output tensor

        """
        self.root = root
        self.s = 1
        self.dlist = [os.path.join(self.root, x) for x in os.listdir(root)]
        self.transform = transform
        self.rotation = np.array([np.linspace(-20, 20, 5)] * self.s).reshape(-1)
        self.CT = []

        # self.drr_win = None
        # self.vis = visdom.Visdom()

        # self.num_samples = len(self.dlist)

        for f in self.dlist:
            path = os.path.join(f, 'xray_256')
            if not os.path.isdir(path):
                os.mkdir(path)
            CT = os.path.join(f, '3d_numpy.npy')

            CT_out = np.load(CT)
            CT_out = np.expand_dims(np.array(CT_out, dtype=np.float32), axis=-1).transpose((3, 2, 1, 0))
            CT_out = torch.tensor(CT_out)
            T = torch.zeros(6, dtype=torch.float32)
            for rot in self.rotation:
                T[2] = torch.tensor(rot)
                drr = utils.DRR_generation(torch.tensor(CT_out), T.view(1, 6), 1)
                if rot < 0:
                    r = '_'+str(int(rot))[1:]
                    np.save(os.path.join(path, r), drr.cpu().numpy())
                else:
                    np.save(os.path.join(path, str(int(rot))), drr.cpu().numpy())


        # im = drr.view((960, 1240)).cpu().numpy()
        # self.drr_win = utils.PlotImage(vis=self.vis, img=im, win=self.drr_win, title="DRR")

        ct_mean = torch.mean(CT_out)
        ct_std = torch.std(CT_out)
        CT_out = (CT_out - ct_mean) / ct_std

    def __getitem__(self, index):
        """

        :param index:
        :return: CT_out: [C, D, H, W] == [1, 393, 512, 512]
        :return drr: [C, H, W]
        :return T : [6]
        """


        return 0
    def __len__(self):
        return self.rotation.size


if __name__ == "__main__":
    # train_path = './registration/2D3D_Data/train'
    # test_path = './registration/2D3D_Data/test'
    train_path = '/home/srk1995/pub/db/Dicom_Image_Unet_pseudo/Train/'
    test_path = '/home/srk1995/pub/db/Dicom_Image_Unet_pseudo/Test/'

    # cTdataloader = Data(root, transform=transforms.ToTensor())
    kdata_train = SegData(train_path, transform=transforms.ToTensor())
    kdata_test = SegData(test_path, transform=transforms.ToTensor())

    # for i, data in enumerate(trainloader):
    #     print(data)

    print("EOP")