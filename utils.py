import numpy as np
import torch.nn as nn
import torch
from torch.nn import functional as F
from scipy.spatial.transform import Rotation as R
import cv2
from scipy.interpolate import RegularGridInterpolator, LinearNDInterpolator, interp1d
import itertools


def ReadText(vis):
    testwindow = vis.text("Hello World!")
    return 0


def PlotImage(vis, img, win, env, title=""):
    # img = img.detach.cpu().numpy()
    win = vis.images(img, win=win, opts=dict(title=title), env=env)
    return win


def PlotLoss(vis, x, y, win, env, legend, title=""):
    if win == None:
        win = vis.line(Y=y, X=x, win=win, opts=dict(title=title, legend=legend, showlegend=True), env=env)
    else:
        win = vis.line(Y=y, X=x, win=win, opts=dict(title=title, legend=legend, showlegend=True), env=env,
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


def cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)


def raycasting(CT, R_pred, num, R_):
    """
    :param CT:
    :param R_pred:
    :param num:
    :param R_:
    :return:
    """

    position = np.loadtxt('PixelPosition' + str(num) + '.txt')
    pixel_spacing = [0.308, 0.308]
    # pixel_spacing = [1, 1]

    ct_pix = [512, 512]
    proj_pix = [960, 1240]

    # Camera matrix
    # R_pred = R_pred.cpu().detach().numpy()
    R_pred = R_.cpu().detach().numpy()
    Rx = R.from_euler('x', R_pred[:, 0], degrees=True)
    Ry = R.from_euler('y', R_pred[:, 1], degrees=True)
    Rz = R.from_euler('z', R_pred[:, 2], degrees=True)
    r = Rx * Ry * Rz

    t = torch.tensor(np.array([[R_pred[:, 3]], [R_pred[:, 4]], [R_pred[:, 5]]]))
    K = torch.tensor(np.loadtxt('intrinsic_mat.txt'), dtype=torch.float32)
    rot = torch.tensor(r.as_dcm(), dtype=torch.float32)

    # Assign image coordinate
    s_min, s_max = -600, -400
    ss = 1
    img_pts = np.array([np.mgrid[1:proj_pix[1]+1, 1:proj_pix[0]+1].T.reshape(-1, 2)] * ((s_max-s_min)//ss))

    # img_pts = img_pts.reshape(-1, 2)
    img_pts = torch.tensor(img_pts).view((-1, 2))

    s = torch.tensor(np.mgrid[s_min:s_max:ss].repeat(proj_pix[0] * proj_pix[1]))
    s = s.view((-1, 1))

    img_pts = torch.cat([img_pts, s], dim=-1).numpy()

    img_pts = img_pts.reshape(((s_max-s_min)//ss, proj_pix[0], proj_pix[1], 3)).transpose((3, 0, 1, 2)).reshape(3, -1, 1)

    img_pts = torch.tensor(np.tile(img_pts, (1, 1, num)).transpose((2, 0, 1)))
    backp = torch.matmul(torch.matmul(torch.inverse(rot), torch.inverse(K)),
                      img_pts - torch.matmul(K, t.view((3, num))).T.reshape((num, 3, 1)))
    backp = backp.view((num, 3, (s_max-s_min)//ss, -1)).permute((0, 3, 2, 1))  # num, -1, 200, 3


    x = np.linspace(-ct_pix[0]/2 * pixel_spacing[0] - position[0][0], ct_pix[0]/2 * pixel_spacing[0] - position[0][0], 512)
    y = np.linspace(-ct_pix[1] / 2 * pixel_spacing[1] - position[0][1],
                    ct_pix[1] / 2 * pixel_spacing[1] - position[0][1], 512)
    z = np.array(position[:, 2])

    tt = cartesian_product(x, y, z)

    ttt = np.matmul(np.matmul(K, rot), tt.T) + np.matmul(K, t.view(3, 1))
    ttt = ttt.T.cpu().numpy()

    min_v = np.array([x[0], y[0], z[0]])
    max_v = np.array([x[-1], y[-1], z[-1]])

    n_backp = (backp - (min_v + max_v)/2) / ((max_v - min_v)/2)

    # Set the distance between camera center and object

    # Backproject to the world coordinate
    # my_interpolating_function = RegularGridInterpolator((x, y, z), np.array(CT[0, 0].cpu()))
    V = CT.cuda()
    V = V.view((num, 1, V.size(1), 512, 512))
    n_backp = torch.tensor(n_backp, dtype=torch.float32).cuda().view((num, proj_pix[0], proj_pix[1], (s_max-s_min)//ss, 3)).permute(0, 3, 1, 2, 4)
    g = torch.nn.functional.grid_sample(V, n_backp, mode='bilinear', padding_mode='border')
    proj_im = torch.sum(g, dim=2)
    # print(proj_im.max())
    return proj_im

def DRR_generation(CT, R_pred, num):
    """
    :param CT:
    :param R_pred:
    :param num:
    :param R_:
    :return:
    """

    ct_pix = [512, 512]
    proj_pix = [256, 256]

    # Camera matrix
    R_pred = R_pred.cpu().detach().numpy()
    # R_pred = R_.cpu().numpy()
    Rx = R.from_euler('x', R_pred[:, 0], degrees=True)
    Ry = R.from_euler('y', R_pred[:, 1], degrees=True)
    Rz = R.from_euler('z', R_pred[:, 2], degrees=True)
    r = Rx * Ry * Rz

    t = torch.tensor(np.array([[R_pred[:, 3]], [R_pred[:, 4]], [R_pred[:, 5]]])).cuda(1)
    f = 0.5
    K = torch.tensor([[f, 0, proj_pix[0]/2], [0, f, proj_pix[1]/2], [0, 0, 1]], dtype=torch.float32).cuda(1)
    rot = torch.tensor(r.as_dcm(), dtype=torch.float32).cuda(1)

    # Assign image coordinate
    s_min, s_max = -150, 150
    ss = 1
    img_pts = np.array([np.mgrid[1:proj_pix[1]+1, 1:proj_pix[0]+1].T.reshape(-1, 2)] * ((s_max-s_min)//ss))

    # img_pts = img_pts.reshape(-1, 2)
    img_pts = torch.tensor(img_pts).view((-1, 2))

    s = torch.tensor(np.mgrid[s_min:s_max:ss].repeat(proj_pix[0] * proj_pix[1]))
    s = s.view((-1, 1))

    img_pts = torch.cat([img_pts, s], dim=-1).numpy()

    img_pts = img_pts.reshape(((s_max-s_min)//ss, proj_pix[0], proj_pix[1], 3)).transpose((3, 0, 1, 2)).reshape(3, -1, 1)

    img_pts = torch.tensor(np.tile(img_pts, (1, 1, num)).transpose((2, 0, 1))).cuda(1)
    backp = torch.matmul(torch.matmul(torch.inverse(rot), torch.inverse(K)),
                      img_pts - torch.matmul(K, t.view((3, num))).T.reshape((num, 3, 1)))
    backp = backp.view((num, 3, (s_max-s_min)//ss, -1)).permute((0, 3, 2, 1))  # num, -1, 200, 3

    # x = np.linspace(-ct_pix[0]/2, ct_pix[0]/2 -1, 512)
    # y = np.linspace(-ct_pix[1] / 2, ct_.cuda()pix[1] / 2 -1, 512)
    # z = np.linspace(-CT.size(1)/2, CT.size(1)/2-1, CT.size(1))
    #
    # tt = cartesian_product(x, y, z)


    min_v = torch.tensor(np.array([-(ct_pix[0]-1)/2, -(ct_pix[1]-1)/2, -(CT.size(1)-1)/2]), dtype=torch.float32).cuda(1)
    max_v = torch.tensor(np.array([(ct_pix[0]-1)/2, (ct_pix[1]-1)/2, (CT.size(1)-1)/2]), dtype=torch.float32).cuda(1)

    n_backp = (backp - (min_v + max_v)/2) / ((max_v - min_v)/2)
    # tt = n_backp.cpu().numpy()

    # Set the distance between camera center and object

    # Backproject to the world coordinate
    V = CT.cuda(1)
    V = V.view((num, V.size(0), V.size(1), V.size(2), V.size(3)))
    n_backp = n_backp.view((1, proj_pix[0], proj_pix[1], (s_max - s_min) // ss, 3)).permute(0, 3, 1, 2, 4)
    # n_backp = torch.tensor(n_backp, dtype=torch.float32).view((1, proj_pix[0], proj_pix[1], (s_max-s_min)//ss, 3)).permute(0, 3, 1, 2, 4)
    g = torch.nn.functional.grid_sample(V, n_backp, mode='bilinear', padding_mode='border')

    proj_im = torch.sum(g, dim=2).view((1, proj_pix[0], proj_pix[1]))
    proj_im_mean = torch.mean(proj_im)
    proj_im_std = torch.std(proj_im)
    proj_im = (proj_im - proj_im_mean) / proj_im_std
    return proj_im