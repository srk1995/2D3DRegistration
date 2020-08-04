import numpy as np
import torch.nn as nn
import torch
from torch.nn import functional as F
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import RegularGridInterpolator, LinearNDInterpolator, interp1d


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

def interp3(x, y, z, v, xi, yi, zi, method='cubic'):
    q = (x, y, z)
    qi = (xi, yi, zi)
    for j in range(3):
        v = interp1d(q[j], v, axis=j, kind=method)(qi[j])
    return v

def raycasting(CT, Xray, R_pred, num):
    """
    :param CT:
    :param Xray:
    :param R_pred:
    :param num:
    :return:
    """

    position = np.loadtxt('PixelPosition' + str(num) + '.txt')
    pixel_spacing = [0.308, 0.308]
    # pixel_spacing = [1, 1]

    ct_pix = [512, 512]
    proj_pix = [960, 1024]
    proj_sz = [1, 1]

    # Camera matrix
    R_pred = R_pred.cpu().detach().numpy()[0]
    Rx = R.from_euler('x', R_pred[0], degrees=True)
    Ry = R.from_euler('y', R_pred[1], degrees=True)
    Rz = R.from_euler('z', R_pred[2], degrees=True)
    r = Rx * Ry * Rz

    t = np.array([[R_pred[3]], [R_pred[4]], [R_pred[5]]])
    extrinsic = np.concatenate((r.as_dcm().transpose(), t), axis=-1)
    K = np.loadtxt('intrinsic_mat.txt')
    rot = r.as_dcm().transpose()


    # Test projection
    pts = np.array([[-1, -1, 512, 1], [-1, 1, 512, 1], [1, -1, 512, 1], [1, 1, 512, 1], [-1, -1, 513, 1], [-1, 1, 513, 1], [1, -1, 513, 1], [1, 1, 513, 1]]).transpose(1, 0)

    pts = np.array([np.mgrid[-ct_pix[0]/2 * pixel_spacing[0] - position[0][0]:ct_pix[0]/2 * pixel_spacing[0] - position[0][0]:pixel_spacing[0], -ct_pix[0]/2 * pixel_spacing[1]- position[0][1]:ct_pix[1]/2 * pixel_spacing[1] - position[0][1]: pixel_spacing[1]].T.reshape(-1, 2)]*position.shape[0])
    pts = pts.reshape(-1, 2)
    z = np.repeat(np.array(position[:, 2]), ct_pix[0]*ct_pix[1])
    z = z.reshape(-1, 1)

    pts = np.hstack((pts, z))
    # projected = np.matmul(np.matmul(K, rot), pts) + np.matmul(K, t)
    # projected_inhomo = projected / projected[2, :]
    #
    # # Assign world coordinate (first: up to down, last: left to right)`
    # tworld_x = np.array(torch.arange(1, ct_pix[0]+1, dtype=torch.float32).repeat(ct_pix[1], 1).transpose(0, 1).reshape(1, -1))[0] - (ct_pix[0]+1)/2
    # # tworld_x = np.repeat(world_x[:, np.newaxis], len(self.position), axis=1)
    # tworld_y = np.flip(np.array(torch.arange(1, ct_pix[1]+1, dtype=torch.float32).repeat(1, ct_pix[0]))[0], 0) - (ct_pix[1]+1)/2
    # # tworld_y = np.repeat(world_y[:, np.newaxis], len(self.position), axis=1)
    # world_x, world_y, world_z = np.zeros_like(tworld_x), np.zeros_like(tworld_x), np.zeros_like(tworld_x)
    # x = np.array(torch.arange(1, ct_pix[0]+1, dtype=torch.float32)) - (ct_pix[0]+1)/2
    # y = np.array(torch.arange(1, ct_pix[1]+1, dtype=torch.float32)) - (ct_pix[1]+1)/2

    # tt = pts.reshape((393, 512, 512, 3))

    # Assign image coordinate
    s_min, s_max = -800, -600
    img_pts = np.array([np.mgrid[1:proj_pix[1]+1, 1:proj_pix[0]+1].T.reshape(-1, 2)] * (s_max-s_min))

    # img_pts = np.hstack((img_pts, np.ones((len(img_pts), 1))))
    img_pts = img_pts.reshape(-1, 2)
    s = np.repeat(np.mgrid[s_min:s_max], proj_pix[0]*proj_pix[1])
    s = s.reshape(-1, 1)

    img_pts = np.hstack((img_pts, s))

    img_pts = img_pts.reshape((s_max-s_min, 960, 1024, 3)).transpose((3, 0, 1, 2)).reshape((3, -1))

    backp = np.matmul(np.matmul(np.linalg.inv(rot), np.linalg.inv(K)), img_pts - np.matmul(K, t))
    backp = backp.reshape((3, s_max-s_min, -1)).transpose((2, 1, 0))  # -1, 200, 3


    x = np.linspace(-ct_pix[0]/2 * pixel_spacing[0] - position[0][0], ct_pix[0]/2 * pixel_spacing[0] - position[0][0], 512)
    y = np.linspace(-ct_pix[1] / 2 * pixel_spacing[1] - position[0][1],
                    ct_pix[1] / 2 * pixel_spacing[1] - position[0][1], 512)
    z = np.array(position[:, 2])
    # Set the distance between camera center and object

    # Backproject to the world coordinate
    my_interpolating_function = RegularGridInterpolator((x, y, z), np.array(CT[0, 0].cpu()))
    # g = LinearNDInterpolator(pts, CT[0].numpy().ravel())

    tt = my_interpolating_function(backp)
    xx, yy, zz = np.meshgrid(x, y, z)
    xx, yy, zz = xx.flatten(), yy.flatten(), zz.flatten()
    xx, yy, zz = np.append(xx, [500]), np.append(yy, [500]), np.append(zz, [500])
    V = np.array(CT[0, 0].cpu()).flatten()
    V = np.append(V, [0])
    proj_im = np.zeros(len(timg_x))
    result = 0
    interpolate = LinearNDInterpolator(np.array([xx, yy, zz]).T, V)
    for k in range(len(s)):
        # pytorch index + array = sum(index)
        X0 = s[k] * np.matmul(np.linalg.inv(R_), np.squeeze(np.array([[timg_x], [timg_y], [np.ones_like(timg_x)]]))) - np.matmul(
            np.linalg.inv(R_), T_)
        img_X = np.array(
            [X0[0] / spacing[0], X0[1] / spacing[1], X0[2]])
        check=((img_X[0] < min(x)) | (img_X[0] > max(x))) | ((img_X[1] < min(y)) | (img_X[1] > max(y))) | ((img_X[2] < min(z)) | (img_X[2] > max(z)))
        img_X = np.where(check == True, 500, img_X)

        result += interpolate(img_X)
            # result += my_interpolating_function(np.transpose(img_X))
        # if result != 0:
            # print(result)
    proj_im = result

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