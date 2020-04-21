import numpy as np
import torch.nn as nn
import torch
from torch.nn import functional as F
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import RegularGridInterpolator


def ReadText(vis):
    testwindow = vis.text("Hello World!")
    return 0

def PlotImage(vis, img, win, title=""):
    # img = img.detach.cpu().numpy()
    win = vis.images(img, win=win, opts=dict(title=title))
    return win

def PlotLoss(vis, x, y, win, title=""):
    if win == None:
        win = vis.line(Y=y, X=x, win=win, opts=dict(title=title, showlegend=True))
    else:
        win = vis.line(Y=y, X=x, win=win, opts=dict(title=title, showlegend=True),
                       update='append')
        # win = vis.line(Y=y, X=x, win=win, opts=dict(title=title, legend=['Train', 'Validation'], showlegend=True), update='append')
    return win

def crop_image(image, label, j):
    sz = image.size()
    x = [x for x in range(sz[2] //128)]
    y = [y for y in range(sz[3] //128)]
    x = np.repeat(np.tile(x, (1, sz[2] //128)).reshape((-1)), image.size()[-1]//32 + 1)
    y = np.repeat(y, sz[3] //128 * (image.size()[-1]//32 + 1))
    z = [z for z in range(image.size()[-1]//32 + 1)]
    z = np.tile(z, (1, sz[2] //128 * sz[3] //128)).reshape((-1))
    if j % (image.size()[-1]//32 + 1) == image.size()[-1]//32:
        img = image[:, :, x[j] * 128:(x[j] + 1) * 128, y[j] * 128:(y[j] + 1) * 128, -32:]
        lb = label[:, :, x[j] * 128:(x[j] + 1) * 128, y[j] * 128:(y[j] + 1) * 128, -32:]
    else:
        img = image[:, :, x[j] * 128:(x[j] + 1) * 128, y[j] * 128:(y[j] + 1) * 128, z[j] * 32:(z[j] + 1) * 32]
        lb = label[:, :, x[j] * 128:(x[j] + 1) * 128, y[j] * 128:(y[j] + 1) * 128, z[j] * 32:(z[j] + 1) * 32]
    return img, lb

def normalization(input):
    min = input.min()
    input = input - min
    max = input.max()
    output = input / max
    return output

def standardization(input):
    mean = input.mean()
    std = torch.std(input)
    input = input - mean
    output = input/std
    return output


def CE(output, target, weights):
    nll = nn.NLLLoss(weight=torch.Tensor([1, 7500]).float())
    return nll(output, target)

def dice_loss(true, logits, eps=1e-7):
    """Computes the SørensenDice loss.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the dice loss so we
    return the negated dice loss.
    Args:
        true: a tensor of shape [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.
    Returns:
        dice_loss: the SørensenDice loss.
    """
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[torch.tensor(true.squeeze(1), dtype=torch.long)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    dice_loss = (2. * intersection / (cardinality + eps)).mean()
    return (1 - dice_loss)


def raycasting(CT, Xray, R_pred, num):

    position = np.loadtxt('PixelPosition' + str(num) + '.txt')
    # pixel_spacing = [0.308, 0.308]
    pixel_spacing = [1, 1]

    ct_pix = [512, 512]
    proj_pix = [960, 1024]
    proj_sz = [1, 1]

    # Camera matrix
    R_pred = R_pred.cpu().detach().numpy()[0]
    Rx = R.from_euler('x', R_pred[0], degrees=True)
    Ry = R.from_euler('y', R_pred[1], degrees=True)
    Rz = R.from_euler('z', R_pred[2], degrees=True)
    r = Rx * Ry * Rz

    camloc = np.array([[R_pred[3]], [R_pred[4]], [R_pred[5]]])
    extrinsic = np.concatenate((r.as_dcm().transpose(), -camloc), axis=-1)
    # intrinsic = [[f, 0, proj_pix[0]/2], [0, f, proj_pix[1]/2], [0, 0, 1]]
    intrinsic = np.loadtxt('intrinsic_mat.txt')
    cam_mat = np.matmul(intrinsic, extrinsic)

    R_ = cam_mat[:, :3]
    # R_ = [[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0]]
    T_ = np.transpose(cam_mat[:, 3][np.newaxis])


    # Test projection
    RT = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]
    tcam_mat = np.matmul(intrinsic, RT)
    pts = np.array([[-1, -1, 512, 1], [-1, 1, 512, 1], [1, -1, 512, 1], [1, 1, 512, 1], [-1, -1, 513, 1], [-1, 1, 513, 1], [1, -1, 513, 1], [1, 1, 513, 1]]).transpose(1, 0)

    projected = np.matmul(tcam_mat, pts)
    projected_inhomo = projected / projected[2, :]

    # Assign world coordinate (first: up to down, last: left to right)`
    tworld_x = np.array(torch.arange(1, ct_pix[0]+1, dtype=torch.float32).repeat(ct_pix[1], 1).transpose(0, 1).reshape(1, -1))[0] - (ct_pix[0]+1)/2
    # tworld_x = np.repeat(world_x[:, np.newaxis], len(self.position), axis=1)
    tworld_y = np.flip(np.array(torch.arange(1, ct_pix[1]+1, dtype=torch.float32).repeat(1, ct_pix[0]))[0], 0) - (ct_pix[1]+1)/2
    # tworld_y = np.repeat(world_y[:, np.newaxis], len(self.position), axis=1)
    world_x, world_y, world_z = np.zeros_like(tworld_x), np.zeros_like(tworld_x), np.zeros_like(tworld_x)
    x = np.array(torch.arange(1, ct_pix[0]+1, dtype=torch.float32)) - (ct_pix[0]+1)/2
    y = np.array(torch.arange(1, ct_pix[1]+1, dtype=torch.float32)) - (ct_pix[1]+1)/2
    z = np.array(position[:, 2])

    Proj_x = np.repeat(world_x[:, np.newaxis], len(position), axis=1)
    Proj_y = np.repeat(world_y[:, np.newaxis], len(position), axis=1)
    s = np.repeat(world_y[:, np.newaxis], len(position), axis=1)
    spacing = [pixel_spacing[0], pixel_spacing[1]]
    for i in range(len(position)):
        world_x = tworld_x * pixel_spacing[0]
        world_y = tworld_y * pixel_spacing[1]
        c = np.transpose(position[i])
        world_x += c[0]
        world_y += c[1]
        world_z += c[2]

        inhomo_img = np.matmul(cam_mat, np.array([[world_x], [world_y], [world_z], [1]]))
        homo_img = inhomo_img[:2] / inhomo_img[2]
        Proj_x[:, i] = homo_img[0][0]
        Proj_y[:, i] = homo_img[1][0]
        s[:, i] = inhomo_img[2][0]
        # X0 = s*np.matmul(np.linalg.inv(R_), np.array([[world_x[i]], [world_y[i]], [1]])) - np.matmul(np.linalg.inv(R_), T_)
        # print("World x: {}, World y:{}, homo x: {}, homo y: {}\n".format(world_x[j], world_y[j], homo_img[0], homo_img[1]))



    # Assign image coordinate
    timg_x = np.array(torch.arange(1, proj_pix[0]+1, dtype=torch.float32).repeat(proj_pix[1], 1).transpose(0, 1).reshape(1, -1))[0] - proj_pix[0]/2
    timg_y = np.flip(np.array(torch.arange(1, proj_pix[1]+1, dtype=torch.float32).repeat(1, proj_pix[0]))[0], 0) - proj_pix[1]/2
    timg_x /= proj_pix[0] / proj_sz[0]
    timg_y /= proj_pix[1] / proj_sz[1]

    # Set the distance between camera center and object
    s = np.arange(-600, -400, 2)

    # Backproject to the world coordinate
    my_interpolating_function = RegularGridInterpolator((x, y, z), np.array(CT[0, 0].cpu()))
    proj_im = np.zeros(len(timg_x))
    for j in range(len(timg_x)):
        result = 0
        for k in range(len(s)):
            # pytorch index + array = sum(index)
            X0 = s[k] * np.matmul(np.linalg.inv(R_), np.array([[timg_x[j]], [timg_y[j]], [1]])) - np.matmul(
                np.linalg.inv(R_), T_)
            img_X = np.array(
                [X0[0] / spacing[0], X0[1] / spacing[1], X0[2]])
            if (img_X[0] < min(x) or img_X[0] > max(x)) or (img_X[1] < min(y) or img_X[1] > max(y)) or (
                    img_X[2] < min(z) or img_X[2] > max(z)):
                result += 0
            else:
                result += my_interpolating_function(np.transpose(img_X))
            # if result != 0:
                # print(result)
        proj_im[j] = result

    # pix = self.pixel_spacing.view(sz(1), 2, 1).repeat(1, 1, 1024*1024)
    # coord = inhomo_coord.type(torch.DoubleTensor) * pix.type(torch.DoubleTensor)
    # coord_xy = torch.cat((coord.type(torch.FloatTensor), torch.zeros_like(coord[:, 0, :].view(sz(1), 1, -1), dtype=torch.float32)), dim=1)
    # coord_xy += self.position.type(torch.FloatTensor).view(sz(1), 3, 1).repeat(1, 1, 1024*1024)
    # one = torch.ones_like(coord_xy[:, 0, :]).view(sz(1), 1, -1)
    # world_coord = torch.cat((coord_xy, one), dim=1).transpose(1, 0).reshape(4, -1).cuda()


    # image_coord = torch.mm(cam_mat, world_coord)
    # image_coord_inhomo = image_coord[:2] / image_coord[2]
    # image_coord_inhomo = image_coord_inhomo.reshape(2, sz(1), -1).transpose(1, 0)
    # projected =

    # print(self.position)


    proj_im = proj_im.reshape((960, 1024))
    return proj_im