from utils import CE, crop_image, dice_loss
from sklearn.metrics import confusion_matrix
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "8"
import ConvNet
import dataloader
import torch.optim as optim
import numpy as np
import torch
import torch.nn as nn
import utils
import visdom
from dataloader import Data
from torchvision import transforms
from torch.utils.data import DataLoader


def transform(img):
    max_value = img.max()
    img_tensor = torch.from_numpy(img.astype(np.int32))
    img_tensor = img_tensor.float() / max_value
    return img_tensor

def custom_loss(output, labels, drr, xray):
    mse = torch.nn.MSELoss()
    loss = mse(output, labels) + alpha * mse(drr, xray)
    return loss


train_root = './registration/2D3D_Data/train'
test_root = './registration/2D3D_Data/test'
PATH = './saved/'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vis = visdom.Visdom()

train_batch_num = 1
s = 4
alpha = 1e-2

# train_win = vis.line(Y=torch.randn(1), X=np.array([5]), opts=dict(title="Train"))
# test_win = vis.line(Y=torch.randn(1), X=np.array([5]), opts=dict(title="Test"))
train_loss_win = None
test_loss_win = None
train_drr_win = None
test_drr_win = None
train_xray_win = None
test_xray_win = None


train_dataset = Data(train_root, transform=transforms.ToTensor())
test_dataset = Data(test_root, transform=transforms.ToTensor())
trainloader = DataLoader(train_dataset, batch_size=train_batch_num, shuffle=True, num_workers=0)
testloader = DataLoader(test_dataset, batch_size=train_batch_num, shuffle=False, num_workers=0)

net = ConvNet.Net_split(1, 16, 6)
net = net.cuda()
net = nn.DataParallel(net)

criterion = torch.nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=1e-2, weight_decay=5e-4)
# train_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)

best_loss = np.inf


def train(net, loader, optimizer, loss_win, drr_win, xray_win):
    train_loss = 0.0
    net.train()

    for i, data in enumerate(loader, 0):
        # inputs and labels.
        inputs = data[0][:, :, ::s, ::s, ::3]
        inputs_X = data[1][:, :, ::s, ::s]
        inputs, inputs_X, labels, num = inputs.to(device), inputs_X.to(device), data[2].to(device), data[3]
        # Set the gradient to be 0.
        optimizer.zero_grad()

        # Train -> Back propagation -> Optimization.
        outputs = net(inputs, inputs_X)

        drr = utils.raycasting(data[0], outputs, data[0].__len__())
        loss = custom_loss(outputs, labels, drr, data[1].cuda())

        loss.backward()
        optimizer.step()

        drr_win = utils.PlotImage(vis=vis, img=drr[0].cpu().numpy().squeeze(), win=drr_win, title="Train DRR")
        xray_win = utils.PlotImage(vis=vis, img=data[1][0].cpu().numpy().squeeze(), win=xray_win, title="Train X-ray")
        if i % 2000 == 0:
            loss_win = utils.PlotLoss(vis=vis, x=torch.Tensor([i]), y=torch.Tensor([loss]), win=loss_win, title="Train Loss")

        train_loss += loss.item()

    return train_loss, loss_win, drr_win, xray_win

def test(net, loader, optimizer, loss_win, drr_win, xray_win):
    test_loss = 0.0
    net.eval()

    for i, data in enumerate(loader, 0):
        # inputs and labels.
        inputs = data[0][:, :, ::s, ::s, ::3]
        inputs_X = data[1][:, :, ::s, ::s]
        inputs, inputs_X, labels, num = inputs.to(device), inputs_X.to(device), data[2].to(device), data[3]
        # Set the gradient to be 0.
        optimizer.zero_grad()

        # Feed forward
        outputs = net(inputs, inputs_X)

        drr = utils.raycasting(data[0], outputs, data[0].__len__())
        loss = custom_loss(outputs, labels, drr, data[1].cuda())

        drr_win = utils.PlotImage(vis=vis, img=drr[0].cpu().numpy().squeeze(), win=drr_win, title="Test DRR")
        xray_win = utils.PlotImage(vis=vis, img=data[1][0].cpu().numpy().squeeze(), win=xray_win, title="Test X-ray")
        if i % 2000 == 0:
            loss_win = utils.PlotLoss(vis=vis, x=torch.Tensor([i]), y=torch.Tensor([loss]), win=loss_win, title="Test Loss")

        test_loss += loss.item()

    return test_loss, loss_win, drr_win, xray_win


if __name__ == "__main__":
    for epoch in range(200):
        train_loss, train_loss_win, train_drr_win, train_xray_win= train(net, testloader, optimizer, train_loss_win, train_drr_win, train_xray_win)
        test_loss, test_loss_win, test_drr_win, test_xray_win = test(net, testloader, optimizer, test_loss_win, test_drr_win, test_xray_win)
        # train_scheduler.step(epoch)

        print('%d train loss: %.3f, test loss: %.3f' % (epoch + 1, train_loss, test_loss))
        is_best = test_loss < best_loss
        best_loss = min(test_loss, best_loss)
        if is_best:
            torch.save({
                'epoch': epoch + 1,
                'state_dict': net.state_dict(),
                'best_loss': best_loss,
                'optimizer': optimizer.state_dict(),
            }, PATH + 'BEST.pth')





    print('Finished Training')