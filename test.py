from utils import CE, crop_image, dice_loss
from sklearn.metrics import confusion_matrix
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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
import matplotlib.pyplot as plt


def transform(img):
    max_value = img.max()
    img_tensor = torch.from_numpy(img.astype(np.int32))
    img_tensor = img_tensor.float() / max_value
    return img_tensor
# transform = transforms.Compose([
#                 transforms.ToTensor()
#                 ])


train_root = './registration/2D3D_Data/train'
test_root = './registration/2D3D_Data/test'
PATH = './saved/'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vis = visdom.Visdom()

train_batch_num = 1

# train_win = vis.line(Y=torch.randn(1), X=np.array([5]), opts=dict(title="Train"))
# test_win = vis.line(Y=torch.randn(1), X=np.array([5]), opts=dict(title="Test"))
train_loss_win = None
train_acc_win = None
test_loss_win = None
test_img_win = None
test_proj_win = None


train_dataset = Data(train_root, transform=transforms.ToTensor())
test_dataset = Data(test_root, transform=transforms.ToTensor())
trainloader = DataLoader(train_dataset, batch_size=train_batch_num, shuffle=True, num_workers=0)
testloader = DataLoader(test_dataset, batch_size=train_batch_num, shuffle=False, num_workers=0)

net = ConvNet.Net_split(1, 16, 6)
net = net.cuda()
net = nn.DataParallel(net)

criterion = torch.nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=1e-3, weight_decay=5e-4)
train_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)

best_loss = np.inf


def train(net, loader, criterion, optimizer, loss_win, acc_win):
    loss = 0.0
    acc = 0.0
    net.train()

    for i, data in enumerate(loader, 0):
        # inputs and labels.
        inputs = data[0][:, :, ::4, ::4, ::3]
        inputs_X = data[1][:, :, ::4, ::4]
        inputs, inputs_X, labels, num = inputs.to(device), inputs_X.to(device), data[2].to(device), data[3]
        # Set the gradient to be 0.
        optimizer.zero_grad()

        # Train -> Back propagation -> Optimization.
        outputs = net(inputs, inputs_X)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Accuracy
        # _, pred = torch.max(outputs.data, 1)
        # total += labels.size(0)
        # correct += (labels == pred).sum().item()
        # acc = 100 * correct / total

        # if i % 20 == 0:
        #     loss_win = utils.PlotLoss(vis=vis, x=torch.Tensor([i]), y=torch.Tensor([loss]), win=loss_win, title="Train Loss")
        #     acc_win = utils.PlotLoss(vis=vis, x=torch.Tensor([i]), y=torch.Tensor([acc]), win=acc_win, title="Train Accuracy")

        loss += loss.item()

    return loss, acc, loss_win, acc_win

def test(net, loader, criterion, optimizer, loss_win, img_win, proj_win):
    loss = 0.0
    acc = 0.0
    net.eval()

    for i, data in enumerate(loader, 0):
        # inputs and labels.
        inputs = data[0][:, :, ::4, ::4, ::3]
        inputs_X = data[1][:, :, ::4, ::4]
        inputs, inputs_X, labels, num = inputs.to(device), inputs_X.to(device), data[2].to(device), data[3]

        # Set the gradient to be 0.
        optimizer.zero_grad()

        # Feed forward
        outputs = net(inputs, inputs_X)
        loss = criterion(outputs, labels)

        proj_img = utils.raycasting(data[0], data[1], outputs, num[0])

        if i % 10 == 0:
            loss_win = utils.PlotLoss(vis=vis, x=torch.Tensor([i]), y=torch.Tensor([loss]), win=loss_win, title="Test Loss")
            img_win = utils.PlotImage(vis=vis, win=img_win, img=inputs_X[0], title="Test img")
            proj_win = utils.PlotImage(vis=vis, win=proj_win, img=proj_img, title="Test projected img")

        loss += loss.item()

    return loss, acc, loss_win, img_win, proj_win


if __name__ == "__main__":
    checkpoint = torch.load(PATH + 'BEST.pth')
    net.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    for epoch in range(1):
        # train_loss, train_acc, train_loss_win, train_acc_win = train(net, trainloader, criterion, optimizer, train_loss_win, train_acc_win)
        test_loss, test_acc, test_loss_win, test_acc_win, test_proj_win = test(net, testloader, criterion, optimizer, test_loss_win,
                                                                test_img_win, test_proj_win)
        train_scheduler.step(epoch)

        print('%d test loss: %.3f' % (epoch + 1, test_loss))









    print('Finished Training')