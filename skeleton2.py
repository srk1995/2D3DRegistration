from skimage.morphology import skeletonize_3d
import numpy as np
import sys
sys.setrecursionlimit(10**4)



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
    data = np.load("/home/whddltkf0889/바탕화면/region_grow/U-net/0b6cfd56422fa8318fb39c4bc6043fbe_00/3d_RG.npy")
    a = mapping(data)
    skel = a.skel

    xyz = np.where(skel == 1)
    sp = np.array([xyz[0][30], xyz[1][30], xyz[2][30]])
    fp = np.array([xyz[0][-1], xyz[1][-1], xyz[2][-1]])

    result = a.get_road(sp, fp)



