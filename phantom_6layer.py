from utils import CE, crop_image, dice_loss
from sklearn.metrics import confusion_matrix
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
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
    loss = alpha * mse(output, labels) + beta * mse(drr, xray)
    # loss = mse(drr, xray)
    return loss


train_root = './registration/2D3D_Data/train'
test_root = './registration/2D3D_Data/test'
PATH = './saved/'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vis = visdom.Visdom()

train_batch_num = 1
alpha = 1
beta = 1e-2


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
train_dataset = Data(train_root, transform=transfroms_)
test_dataset = Data(test_root, transform=transfroms_)
trainloader = DataLoader(train_dataset, batch_size=train_batch_num, shuffle=True, num_workers=0)
testloader = DataLoader(test_dataset, batch_size=train_batch_num, shuffle=False, num_workers=0)

net = ConvNet.layer6Net(1, 8, 6)
net = net.cuda()
net = nn.DataParallel(net)

criterion = torch.nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
# train_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)

best_loss = np.inf


def train(net, loader, optimizer, drr_win, xray_win, env):
    train_loss = 0.0
    net.train()

    for i, data in enumerate(loader, 0):
        # inputs and labels.
        inputs = data[0]
        inputs_X = data[1]
        inputs, inputs_X, labels= inputs.to(device), inputs_X.to(device), data[2].to(device)
        # Set the gradient to be 0.
        optimizer.zero_grad()

        # Train -> Back propagation -> Optimization.
        outputs = net(inputs, inputs_X)

        drr = utils.DRR_generation(data[0].view(1, inputs.shape[2], inputs.shape[3], inputs.shape[4]), outputs, train_batch_num).view((1, 960, 1240))
        loss = custom_loss(outputs, labels, drr, data[1].cuda())

        loss.backward()
        optimizer.step()

        xray_win = utils.PlotImage(vis=vis, img=data[1][0].cpu().numpy().squeeze(), win=xray_win, env=env,
                                   title="Train X-ray")
        drr_win = utils.PlotImage(vis=vis, img=drr[0].cpu().numpy().squeeze(), win=drr_win, env=env, title="Train DRR")

        train_loss += loss.item()

    return train_loss, drr_win, xray_win


def test(net, loader, optimizer, drr_win, xray_win, env):
    test_loss = 0.0
    net.eval()

    for i, data in enumerate(loader, 0):
        # inputs and labels.
        inputs = data[0]
        inputs_X = data[1]
        inputs, inputs_X, labels= inputs.to(device), inputs_X.to(device), data[2].to(device)
        # Set the gradient to be 0.
        optimizer.zero_grad()

        # Feed forward
        outputs = net(inputs, inputs_X)

        drr = utils.DRR_generation(data[0].view(1, inputs.shape[2], inputs.shape[3], inputs.shape[4]), outputs, train_batch_num).view((1, 960, 1240))
        loss = custom_loss(outputs, labels, drr, data[1].cuda())

        xray_win = utils.PlotImage(vis=vis, img=data[1][0].cpu().numpy().squeeze(), win=xray_win, env=env,
                                   title="Test X-ray")
        drr_win = utils.PlotImage(vis=vis, img=drr[0].cpu().numpy().squeeze(), win=drr_win, env=env, title="Test DRR")

        test_loss += loss.item()

    return test_loss, drr_win, xray_win


if __name__ == "__main__":
    env = "phantom_6layer"
    vis.close(env="phantom_6layer")
    for epoch in range(200):
        train_loss, train_drr_win, train_xray_win = train(net, trainloader, optimizer, train_drr_win, train_xray_win, env)
        test_loss, test_drr_win, test_xray_win = test(net, testloader, optimizer, test_drr_win, test_xray_win, env)
        # train_scheduler.step(epoch)

        # train_loss_win = utils.PlotLoss(vis=vis, x=torch.tensor([epoch]), y=torch.tensor([train_loss]), win=train_loss_win, env=env,
        #                           title="Train Loss")
        # test_loss_win = utils.PlotLoss(vis=vis, x=torch.tensor([epoch]), y=torch.tensor([test_loss]), win=test_loss_win, env=env,
        #                           title="Test Loss")

        x = torch.tensor([epoch+1, epoch+1]).view((-1, 2))
        y = torch.tensor([train_loss, test_loss]).view((-1, 2))
        loss_win = utils.PlotLoss(vis=vis, x=x, y=y, win=loss_win, env=env, legend=['Train', 'Test'],
                                  title="Loss")

        print('%d train loss: %.3f, test loss: %.3f' % (epoch + 1, train_loss, test_loss))
        is_best = test_loss < best_loss
        best_loss = min(test_loss, best_loss)
        if is_best:
            torch.save({
                'epoch': epoch + 1,
                'state_dict': net.state_dict(),
                'best_loss': best_loss,
                'optimizer': optimizer.state_dict(),
            }, PATH + 'BEST_phantom_6layer.pth')

    print('Finished Training')