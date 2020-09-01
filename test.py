from utils import CE, crop_image, dice_loss
from sklearn.metrics import confusion_matrix
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
import ConvNet
import dataloader
import torch.optim as optim
import numpy as np
import torch
import torch.nn as nn
import utils
import visdom
from dataloader import SegData
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


def transform(img):
    max_value = img.max()
    img_tensor = torch.from_numpy(img.astype(np.int32))
    img_tensor = img_tensor.float() / max_value
    return img_tensor

train_root = '/home/srk1995/pub/db/Dicom_Image_Unet_pseudo/Train/'
test_root = '/home/srk1995/pub/db/Dicom_Image_Unet_pseudo/Test/'
PATH = './saved/'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vis = visdom.Visdom()

train_batch_num = 1

alpha = 0
mse = torch.nn.MSELoss()

proj_pix = [256, 256]



# train_win = vis.line(Y=torch.randn(1), X=np.array([5]), opts=dict(title="Train"))
# test_win = vis.line(Y=torch.randn(1), X=np.array([5]), opts=dict(title="Test"))
loss_win = None
train_drr_win = None
test_drr_win = None
train_xray_win = None
test_xray_win = None

transfroms_ = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Resize((64, 64))
])
train_dataset = SegData(train_root, transform=transfroms_)
test_dataset = SegData(test_root, transform=transfroms_)
trainloader = DataLoader(train_dataset, batch_size=train_batch_num, shuffle=True, num_workers=0)
testloader = DataLoader(test_dataset, batch_size=train_batch_num, shuffle=False, num_workers=0)

net = ConvNet.layer6Net(1, 20, 6)
net = net.cuda()
net = nn.DataParallel(net)

criterion = torch.nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
# train_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)

best_loss = np.inf



def test(net, loader, optimizer):
    test_tre = 0.0
    num = 0
    net.eval()

    for i, data in enumerate(loader, 0):
        # inputs and labels.
        inputs = data[0]
        inputs_X = data[1]
        inputs, inputs_X, labels= inputs.cuda(), inputs_X.cuda(), data[2].cuda()
        # Set the gradient to be 0.
        optimizer.zero_grad()

        # Feed forward
        outputs = net(inputs, inputs_X)

        drr = utils.DRR_generation(data[0].view(1, inputs.shape[2], inputs.shape[3], inputs.shape[4]), outputs,
                                   train_batch_num).view((1, proj_pix[0], proj_pix[1]))
        tre = utils.TRE(data[0].view(1, inputs.shape[2], inputs.shape[3], inputs.shape[4]), labels, outputs, train_batch_num)

        plt.imsave("./images/test/alpha1e-2/xray"+str(i)+".png",  inputs_X.view((256, 256)).cpu().numpy())
        plt.imsave("./images/test/alpha1e-2/drr" + str(i) + ".png", drr.view((256, 256)).cpu().numpy())

        test_tre += tre.item()
        num += data[0].size(0)

    return test_tre / num, drr


if __name__ == "__main__":
    env = "seg_6layer_alpha_1e-2_lr_1e-2"
    # vis.close(env=env)
    ck = torch.load("./saved/BEST"+env[3:] + ".pth")
    net.load_state_dict(ck['state_dict'])
    optimizer.load_state_dict(ck['optimizer'])
    best_loss = ck['best_loss']

    for epoch in range(1):
        test_tre, test_drr = test(net, testloader, optimizer)

        print('Target Registration Error: %.3f' % (test_tre))



    print('Finished Training')