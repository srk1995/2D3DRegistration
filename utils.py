import numpy as np
import torch.nn as nn
import torch
from torch.nn import functional as F
from scipy.spatial.transform import Rotation as R
import scipy.io as sio

def OnehotEncoding(arr, c):
    ind = arr // c + c // 2
    ind = ind.type(dtype=torch.long)

    out = torch.zeros((c, 6))
    out[ind, [i for i in range(len(arr))]] = 1

    return out

def OnehotDecoding(arr, c):
    _, out = np.where(arr.T == 1)
    out = (out - c // 2) * c

    return torch.tensor(out)

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

# intersection function
def isect_line_plane_v3(p0, p1, p_co, p_no, epsilon=1e-6):
    """
    p0, p1: Define the line.
    p_co, p_no: define the plane:
        p_co Is a point on the plane (plane coordinate).
        p_no Is a normal vector defining the plane direction;
             (does not need to be normalized).

    Return a Vector or None (when the intersection can't be found).
    """

    # # Test
    # p0 = torch.tensor([[0, 0, 0], [0, 0, 0]], dtype=torch.float32).view(2, 3).T
    # p1 = torch.tensor([[0, 0, 1], [1, 2, 3]], dtype=torch.float32).view(2, 3).T
    #
    # p_co = torch.tensor([20, 10, 30], dtype=torch.float32).view(3, 1)
    # p_no = torch.tensor([0, 0, 10], dtype=torch.float32).view(3, 1)

    # Normalize the normal vector of the plane
    n = torch.norm(p_no, dim=0)
    p_no = p_no / n


    # Normalize the direction vector of the line and calculate degree between the normal vector and the direction vector
    u = p1 - p0
    n = torch.norm(u, dim=0)
    u = u / n
    dot = torch.mm(u.T, p_no)

    # idx = np.where(abs(dot.cpu()) > torch.tensor(epsilon))[0]
    # p0 = p0[:, idx]
    # p1 = p1[:, idx]
    # u = p1 - p0
    # n = torch.norm(u, dim=0)
    # u = u / n
    # dot = torch.mm(u.T, p_no)

    # The factor of the point between p0 -> p1 (0 - 1)
    # if 'fac' is between (0 - 1) the point intersects with the segment.
    # Otherwise:
    #  < 0.0: behind p0.
    #  > 1.0: infront of p1.
    w = p0 - p_co
    fac = -torch.mm(w.T, p_no) / dot
    u = u * fac.T
    vec = p0 + u
    # tt = vec.cpu().numpy()
    return vec

# ----------------------
# generic math functions


def dot_v3v3(v0, v1):
    return (
        (v0[:, 0] * v1[:, 0]) +
        (v0[:, 1] * v1[:, 1]) +
        (v0[:, 2] * v1[:, 2])
        )


def len_squared_v3(v0):
    return dot_v3v3(v0, v0)


def mul_v3_fl(v0, f):
    return (
        v0[0] * f,
        v0[1] * f,
        v0[2] * f,
        )


def create_ranges_nd(start, stop, N, endpoint=True):
    if endpoint==1:
        divisor = N-1
    else:
        divisor = N
    steps = (1.0/divisor) * (stop - start)
    return start[...,None] + steps[...,None]*np.arange(N)

def DRR_generation(CT, R_pred, num, proj_pix):
    """
    :param CT:
    :param R_pred:
    :param num:
    :param R_:
    :return:
    """

    ct_pix = [512, 512]

    min_v = torch.tensor(np.array([-(ct_pix[0]-1)/2, -(ct_pix[1]-1)/2, -(CT.size(1)-1)/2]), dtype=torch.float32).cuda(1)
    max_v = torch.tensor(np.array([(ct_pix[0]-1)/2, (ct_pix[1]-1)/2, (CT.size(1)-1)/2]), dtype=torch.float32).cuda(1)

    # Camera matrix
    R_pred = R_pred.cpu().detach().numpy()
    # R_pred = np.array([[15, -15, 0, 0, 0, 0]], dtype=np.float32)
    # R_pred = R_.cpu().numpy()
    Rx = R.from_euler('x', -R_pred[:, 0], degrees=True)
    Ry = R.from_euler('y', -R_pred[:, 1], degrees=True)
    Rz = R.from_euler('z', -R_pred[:, 2], degrees=True)
    r = Rx * Ry * Rz

    O = torch.tensor([0, 0, -160], dtype=torch.float32).view(3, 1, 1).cuda(1)
    t = -O - torch.tensor(np.array([[R_pred[:, 3]], [R_pred[:, 4]], [R_pred[:, 5]]])).cuda(1)
    # t = (t - (min_v.reshape(3, 1, 1) + max_v.reshape(3, 1, 1))/2) / ((max_v.reshape(3, 1, 1) - min_v.reshape(3, 1, 1))/2)
    f = 256
    n = 200
    K = torch.tensor([[f, 0, proj_pix[0]/2], [0, f, proj_pix[1]/2], [0, 0, 1]], dtype=torch.float32).cuda(1)
    rot = torch.tensor(r.as_dcm(), dtype=torch.float32).cuda(1)


    ## For visualization (1)
    # s_min, s_max = 0, 200
    # ss = 1
    # img_pts = np.array([np.mgrid[1:proj_pix[1]+1, 1:proj_pix[0]+1].T.reshape(-1, 2)] * int(((s_max-s_min)/ss)))
    # img_pts = torch.tensor(img_pts, dtype=torch.float32).view((-1, 2))
    # s = torch.tensor(np.mgrid[s_min:s_max:ss].repeat(proj_pix[0] * proj_pix[1]), dtype=torch.float32)
    # s = s.view((-1, 1))
    # img_pts = torch.cat([img_pts*s, s], dim=-1).numpy()
    # img_pts = img_pts.reshape((int((s_max - s_min) / ss), proj_pix[0], proj_pix[1], 3)).transpose((3, 0, 1, 2)).reshape(
    #     3, -1, 1)
    # img_pts = torch.tensor(np.tile(img_pts, (1, 1, num)).transpose((2, 0, 1))).cuda(1)
    # backp = torch.matmul(torch.matmul(torch.inverse(rot), torch.inverse(K)),
    #                      img_pts - torch.matmul(K, t.view((3, num))).T.reshape((num, 3, 1)))
    # backp = backp.view((num, 3, int((s_max - s_min) / ss), -1)).permute((0, 3, 2, 1))  # num, -1, 200, 3


    ## Original Code (2)
    img_pts = np.array([np.mgrid[1:proj_pix[1] + 1, 1:proj_pix[0] + 1].T.reshape(-1, 2)] * 2)
    img_pts = torch.tensor(img_pts, dtype=torch.float32).view((-1, 2))
    s = torch.tensor(np.mgrid[0:2:1].repeat(proj_pix[0] * proj_pix[1]), dtype=torch.float32)
    s = s.view((-1, 1))
    img_pts = torch.cat([img_pts*s, s], dim=-1).numpy()
    img_pts = img_pts.reshape((2, proj_pix[0], proj_pix[1], 3)).transpose((3, 0, 1, 2)).reshape(3, -1, 1)
    img_pts = torch.tensor(np.tile(img_pts, (1, 1, num)).transpose((2, 0, 1))).cuda(1)
    backp = torch.matmul(torch.matmul(torch.inverse(rot), torch.inverse(K)),
                      img_pts - torch.matmul(K, t.view((3, num))).T.reshape((num, 3, 1)))
    backp = backp.view((num, 3, 2, -1)).permute((0, 3, 2, 1))  # num, -1, 200, 3


    # x = np.linspace(-ct_pix[0]/2, ct_pix[0]/2 -1, 512)
    # y = np.linspace(-ct_pix[1] / 2, ct_.cuda()pix[1] / 2 -1, 512)
    # z = np.linspace(-CT.size(1)/2, CT.size(1)/2-1, CT.size(1))
    #
    # tt = cartesian_product(x, y, z)

    normals = torch.tensor([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=torch.float32).cuda(1)
    pts = normals

    n_backp = (backp - (min_v + max_v) / 2) / ((max_v - min_v) / 2)
    t = -(t.view((3)) - (min_v + max_v) / 2) / ((max_v - min_v) / 2)

    sio.savemat('CT_ray.mat', {'CT': CT.numpy(), 'ray': backp.cpu().numpy()})

    zz = torch.tensor([-(CT.size(1)-1)/2, (CT.size(1)-1)/2], dtype=torch.float32).cuda(1)
    zz = (zz - (min_v[2] + max_v[2]) / 2) / ((max_v[2] - min_v[2]) / 2)
    # p0 = t.view((3, 1)).repeat(1, proj_pix[0] * proj_pix[1])
    p0 = n_backp[:, :, 0, :].squeeze().transpose(1, 0)
    p1 = n_backp[:, :, 1, :].squeeze().transpose(1, 0)
    itsc_list = []
    for i in range(6):
        itsc_list.append(isect_line_plane_v3(p0, p1, pts[i, :].view(3, 1), normals[i, :].view(3, 1), epsilon=1e-6))

    itsc = torch.stack([itsc_list[0], itsc_list[1], itsc_list[2], itsc_list[3], itsc_list[4], itsc_list[5]])
    itsc = itsc.permute(0, 2, 1)
    idx = ((itsc[:, :, 0] <= 1) & (itsc[:, :, 0] >= -1) & (itsc[:, :, 1] <= 1) & (itsc[:, :, 1] >= -1) & (itsc[:, :, 2] <= zz[1]) & (
                itsc[:, :, 2] >= zz[0]))
    # vec = itsc[idx, :].reshape((2, -1, 3)).cpu().numpy()
    z = torch.tensor([[False, False, False, False, True, True]]).reshape(6, 1).repeat(1, idx.size(1)).cuda(1)
    idx = torch.where((torch.sum(idx, dim=0) == 2), idx, z).permute(1, 0)
    itsc = itsc.permute(1, 0, 2)
    vec = itsc[idx, :].view(-1, 2, 3).permute(1, 0, 2).cpu().numpy()
    # vec = itsc[idx, :].view(2, -1, 3).cpu().numpy()

    # Sampling n points between two intersection points
    n_backp = torch.tensor(create_ranges_nd(vec[1], vec[0], n), dtype=torch.float32).cuda(1)


    # n_backp = backp
    # sio.savemat('CT_ray.mat', {'CT': CT.numpy(), 'ray': n_backp.cpu().numpy()})
    # tt = n_backp.cpu().numpy().squeeze()

    # Set the distance between camera center and object

    # Backproject to the world coordinate
    V = CT.cuda(1)
    V = V.view((num, V.size(0), V.size(1), V.size(2), V.size(3)))
    n_backp = n_backp.view((1, proj_pix[0], proj_pix[1], 3, n)).permute(0, 4, 1, 2, 3)  # B x num x 256 x 256 x 3
    # n_backp = torch.tensor(n_backp, dtype=torch.float32).view((1, proj_pix[0], proj_pix[1], (s_max-s_min)//ss, 3)).permute(0, 3, 1, 2, 4)
    g = torch.nn.functional.grid_sample(V, n_backp, mode='bilinear', padding_mode='border')

    proj_im = torch.sum(g, dim=2).view((1, proj_pix[0], proj_pix[1]))
    proj_im_mean = torch.mean(proj_im)
    proj_im_std = torch.std(proj_im)

    if proj_im_std.item() == 0:
        proj_im = torch.zeros_like(proj_im)
        # print(R_pred)
    else:
        proj_im = (proj_im - proj_im_mean) / proj_im_std
    return proj_im


def TRE(CT, R_gt, R_pred, num):
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

    rot_hat = torch.tensor(r.as_dcm(), dtype=torch.float32)
    t_hat = torch.tensor(np.array([[R_pred[:, 3]], [R_pred[:, 4]], [R_pred[:, 5]]]))

    R_gt = R_gt.cpu().detach().numpy()
    Rx = R.from_euler('x', R_gt[:, 0], degrees=True)
    Ry = R.from_euler('y', R_gt[:, 1], degrees=True)
    Rz = R.from_euler('z', R_gt[:, 2], degrees=True)
    r = Rx * Ry * Rz

    rot = torch.tensor(r.as_dcm(), dtype=torch.float32)
    t = torch.tensor(np.array([[R_gt[:, 3]], [R_gt[:, 4]], [R_gt[:, 5]]]))


    f = 0.5
    K = torch.tensor([[f, 0, proj_pix[0]/2], [0, f, proj_pix[1]/2], [0, 0, 1]], dtype=torch.float32)


    x = np.linspace(-ct_pix[0]/2, ct_pix[0]/2 -1, 512)
    y = np.linspace(-ct_pix[1] / 2, ct_pix[1] / 2 -1, 512)
    z = np.linspace(-CT.size(1)/2, CT.size(1)/2-1, CT.size(1))

    tt = cartesian_product(x, y, z)

    X = torch.matmul(torch.matmul(K, rot), torch.tensor(tt, dtype=torch.float32).T) + torch.matmul(K, t.view(
        (3, num)))
    X = X[:, :2, :] / X[:, 2, :]

    X_hat = torch.matmul(torch.matmul(K, rot_hat), torch.tensor(tt, dtype=torch.float32).T) + torch.matmul(K, t_hat.view(
        (3, num)))

    X_hat = X_hat[:, :2, :] / X_hat[:, 2, :]

    abs_v = torch.abs(X - X_hat)
    ind = (torch.isnan(abs_v) ).nonzero()
    ind_f = (torch.isinf(abs_v)).nonzero()
    for i in range(len(ind)):
        abs_v[ind[i][0], ind[i][1], ind[i][2]] = 0

    for i in range(len(ind_f)):
        abs_v[ind_f[i][0], ind_f[i][1], ind_f[i][2]] = 0
    t_s = torch.sum(abs_v)
    total_l = abs_v.size(2) - len(ind) - len(ind_f)
    tre = t_s / total_l

    return tre