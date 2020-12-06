from skimage.morphology import skeletonize_3d
import numpy as np


class mapping():
    def __init__(self, mask):
        self.mask = mask
        self.skel = skeletonize_3d(self.mask)
        self.skel = self.skel / self.skel.max()
        self.x_size = mask.shape[0]
        self.y_size = mask.shape[1]
        self.z_size = mask.shape[2]
        self.stack = []
        k = []
        for x in range(-5, 6):
            for y in range(-5, 6):
                for z in range(-5, 6):
                    if (x ** 2 + y ** 2 + z ** 2 < 4) & ~((x == 0) & (y == 0) & (z == 0)):
                        k.append([x, y, z])

        self.vt = k


    def get_road(self, sp, fp):
        path = []

        max_d = 0
        temp_skel = np.zeros_like(self.skel, dtype=np.int32)
        x = np.array(np.where(self.skel == 1)).transpose([1, 0])
        path_points = np.array([sp])

        while True:
            new_path_points = np.array([], dtype=np.int64)
            for i, (path) in enumerate(self.vt):
                paths = path + path_points
                tf = (self.skel[paths[:, 0], paths[:, 1], paths[:, 2]] == 1) & (
                            temp_skel[paths[:, 0], paths[:, 1], paths[:, 2]] == 0)
                paths = paths[tf]
                if paths.shape[0]:
                    temp_skel[paths[:,0],paths[:,1],paths[:,2]] = i+1
                    if new_path_points.shape[0]:
                        new_path_points = np.append(new_path_points, paths, 0)
                    else:
                        new_path_points = paths
            if new_path_points.shape[0] == 0:
                break
            path_points = new_path_points
        result = np.array([fp], dtype=np.int32)
        p = fp
        while not (p == sp).all():
            p = p - self.vt[temp_skel[p[0],p[1],p[2]] - 1]
            result = np.append([p], result, 0)
        l_m, sr, f = self.connect(sp, fp, 0,[])

        return sr


    def connect(self, point, fp, len_move, stack_road):
        min_lm = len_move

        return_f = 0
        for m in self.vt:
            tf = (point[0] + m[0] < 0) or (point[1] + m[1] < 0) or (point[2] + m[2] < 0) or (point[0] + m[0] >= self.x_size) or (point[1] + m[1] >= self.y_size) or (point[2] + m[2] >= self.z_size)
            if tf:
                pass
            else:
                move_point = point + m
                if (move_point[0] == fp[0]) and (move_point[1] == fp[1]) and (move_point[2] == fp[2]):
                    stack_road.append(list(move_point))
                    return len_move, stack_road, 1
                data = self.skel[point[0] + m[0], point[1] + m[1], point[2] + m[2]]

                if (data == 1) and not (list(move_point) in self.stack):
                    self.stack.append(list(move_point))
                    stack_road.append(list(move_point))
                    l_m, sr, f = self.connect(point + m, fp, len_move + 1, stack_road)

                    if f == 1:
                        return l_m, sr, f
                    else:
                        stack_road.pop()

        return min_lm, stack_road, return_f


if __name__ == "__main__":
    data = np.load("/home/whddltkf0889/바탕화면/region_grow/U-net2/0b6cfd56422fa8318fb39c4bc6043fbe_00/new_3d_RG_npy.npy")

    a = mapping(data)
    xyz = np.where(a.skel == 1)
    sp = np.array([xyz[0][2], xyz[1][2], xyz[2][2]])
    fp = np.array([xyz[0][10000], xyz[1][10000], xyz[2][10000]])
    result = a.get_road(sp, fp)



