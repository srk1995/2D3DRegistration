from utils import CE, crop_image, dice_loss
from sklearn.metrics import confusion_matrix
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import ConvNet
import dataloader
import torch.optim as optim
import numpy as np
import torch
import torch.nn as nn
import utils
import visdom
from dataloader import SegData_csv, SegData_catheter_pt
from torchvision import transforms
from torch.utils.data import DataLoader
import cv2
import argparse
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R


def transform(img):
    max_value = img.max()
    img_tensor = torch.from_numpy(img.astype(np.int32))
    img_tensor = img_tensor.float() / max_value
    return img_tensor

def test(net, loader, optimizer, drr_win, xray_win, env):
    t_loss = 0.0
    r_loss = 0.0
    num = 0
    net.eval()
    lb = utils.OnehotDecoding(np.repeat(np.array([i for i in range(args.qt)]), 1).reshape(-1, 1), val, args.qt)

    for i, data in enumerate(loader, 0):
        # inputs and labels.
        inputs = data[0]
        inputs_X = data[1]
        # CT_v = data[3]
        inputs, inputs_X, labels= inputs.cuda(), inputs_X.cuda(), data[2].cuda()
        # Set the gradient to be 0.
        optimizer.zero_grad()

        # Feed forward
        if (torch.sum(inputs_X) != 0) and (inputs_X.size()[2] > 128):
            outputs = net(inputs, inputs_X)
            _, pred = torch.max(outputs, 1)
            T_pred = utils.OnehotDecoding(pred.cpu(), val, args.qt).float()
            #
            # drr = utils.DRR_generation(CT_v.view(1, CT_v.shape[2], CT_v.shape[3], CT_v.shape[4]), T_pred, train_batch_num, proj_pix).view((1, proj_pix[0], proj_pix[1]))
            # loss = bce(outputs, labels) + mse(drr.cuda(), inputs_X)
            _, labels = torch.max(labels, 1)
            labels = labels.cpu().detach().numpy()

            outputs = torch.sum(outputs * torch.tensor(lb).cuda(), 1).cpu().detach().numpy()
            labels = utils.OnehotDecoding(labels, val, args.qt)

            loss = np.linalg.norm(outputs[0][3:]-labels[0][3:])

            Rx = R.from_euler('x', outputs[:, 0], degrees=True)
            Ry = R.from_euler('y', outputs[:, 1], degrees=True)
            Rz = R.from_euler('z', outputs[:, 2], degrees=True)
            r = Rx * Ry * Rz
            r = r.as_dcm().squeeze()

            # Rx = R.from_euler('x', labels[:, 0], degrees=True)
            # Ry = R.from_euler('y', labels[:, 1], degrees=True)
            Rz = R.from_euler('z', labels[:, 2], degrees=True)
            # r_t = Rx * Ry * Rz
            # r_t = r_t.as_dcm().squeeze()

            r_t = Rz.as_dcm().squeeze()

            rr = np.matmul(r_t, r.T)
            # r = R.from_matrix(rr)
            tr_r = np.trace(rr)

            theta = np.arccos((tr_r -1)/2)
            r_loss += theta


            # tt = drr[0].cpu().numpy().squeeze()
            # tt2 = data[4][0][0].cpu().numpy().squeeze()
            # if (tt.max() != tt.min()):
            #     tt = (tt - tt.min()) / (tt.max() - tt.min())


            # plt.subplot(1, 2, 1)
            # plt.imshow(tt2)
            # plt.subplot(1, 2, 2)
            # plt.imshow(tt)
            # plt.pause(1)


            # xray_win = utils.PlotImage(vis=vis, img=data[1][0].cpu().numpy().squeeze(), win=xray_win, env=env,
            #                            title="Test X-ray")
            # drr_win = utils.PlotImage(vis=vis, img=drr[0].cpu().numpy().squeeze(), win=drr_win, env=env, title="Test DRR")

            t_loss += loss
            num += data[0].size(0)

    return r_loss / num, t_loss / num, drr_win, xray_win


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preocess some numbers.")
    parser.add_argument('--net', type=str, help='Network architecture, 6layer, 8layer, unet, homo, homo_bn', default='pointnet2')
    parser.add_argument('--alpha', type=float, help='alpha', default=1e-4)
    parser.add_argument('--lr', type=float, help='Learning rate', default=1e-3)
    parser.add_argument('--gpu', type=str, help='gpu number', default='1,0')
    parser.add_argument('--qt', type=int, help='The number of bins', default=1024)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    env = "seg_" + args.net + "_alpha_" + str(args.alpha) + "_lr_" + str(args.lr) + "_qt_" + str(args.qt)

    test_file = './test_zz.csv'
    PATH = './saved/'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vis = visdom.Visdom()

    train_batch_num = 1
    alpha = args.alpha

    proj_pix = [256, 256]
    val = [-30, 30]

    bce = torch.nn.BCELoss()
    mse = torch.nn.MSELoss()

    # train_win = vis.line(Y=torch.randn(1), X=np.array([5]), opts=dict(title="Train"))
    # test_win = vis.line(Y=torch.randn(1), X=np.array([5]), opts=dict(title="Test"))
    loss_win = None
    test_drr_win = None
    test_xray_win = None

    transfroms_ = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Resize((64, 64))
    ])
    test_dataset = SegData_catheter_pt(test_file, proj_pix, args.qt, transform=transfroms_)
    testloader = DataLoader(test_dataset, batch_size=train_batch_num, shuffle=True, num_workers=0)
    if args.net == '6layer':
        net = ConvNet.layer6Net(1, 20, 6)
    elif args.net == '8layer':
        net = ConvNet.layer8Net(1, 20, 6)
    elif args.net == 'homo':
        net = ConvNet.HomographyNet(1, 20, 6)
    elif args.net == 'homo_bn':
        net = ConvNet.HomographyNet_bn(1, 20, 6)
    elif args.net == 'pointnet2':
        net = ConvNet.PointReg(1 * args.qt, False)
    else:
        net = ConvNet.UNet(1, 20, 6)

    net = net.cuda()
    net = nn.DataParallel(net)

    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)
    # train_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)

    best_loss = np.inf

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # vis.close(env="seg_6layer")
    if os.path.isfile("./saved/BEST" + env[3:] + "1.pth"):
        ck = torch.load("./saved/BEST" + env[3:] + "1.pth")
        net.load_state_dict(ck['state_dict'])
        optimizer.load_state_dict(ck['optimizer'])
        start = ck['epoch']
        best_loss = ck['best_loss']
    for epoch in range(1):
        r_loss, t_loss, test_drr_win, test_xray_win = test(net, testloader, optimizer, test_drr_win, test_xray_win, env)
        scheduler.step()
        # train_scheduler.step(epoch)

        # train_loss_win = utils.PlotLoss(vis=vis, x=torch.tensor([epoch]), y=torch.tensor([train_loss]), win=train_loss_win, env=env,
        #                           title="Train Loss")
        # test_loss_win = utils.PlotLoss(vis=vis, x=torch.tensor([epoch]), y=torch.tensor([test_loss]), win=test_loss_win, env=env,
        #                           title="Test Loss")

        # x = torch.tensor([epoch + 1, epoch + 1]).view((-1, 2))
        # loss_win = utils.PlotLoss(vis=vis, x=x, y=y, win=loss_win, env=env, legend=['Train', 'Test'],
        #                           title="Loss")

        print("Rotation loss (rad): {}".format(r_loss))
        print("Translation loss (pixel): {}".format(t_loss))

    print('EOP')