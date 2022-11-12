from corner_detection import CornerDetection
from homography import Homography

import numpy as np
import itertools


class Zhang:
    def __init__(self, data_dir="Dataset1") -> None:
        self.cd = CornerDetection(data_dir=data_dir)
        self.hm = Homography()
        # generate actual points for computing homography
        x = np.arange(0, 160, 20)
        y = np.arange(0, 200, 20)
        actual_pts = list(itertools.product(x, y))
        self.actual_pts_aug = [list(x) + [1] for x in actual_pts]

    def get_lines_pts_H(self, img_list):
        v_lines_all, h_lines_all, pts_all, H_all = [], [], [], []
        valid_images = []
        for i in img_list:
            img = self.cd.images[i]
            lines, h_lines, v_lines = self.cd.get_lines(img)
            v_lines_sorted, h_lines_sorted = self.cd.get_sorted_lines(v_lines, h_lines)
            pts = self.cd.get_intersection_points(v_lines_sorted, h_lines_sorted)
            if len(pts) == 80:
                A = self.hm.get_A(self.actual_pts_aug, pts)
                b = self.hm.get_B(pts)
                H = self.hm.get_H(A, b)

                v_lines_all.append(v_lines_sorted)
                h_lines_all.append(h_lines_sorted)
                pts_all.append(pts)
                H_all.append(H)
                valid_images.append(i)
        return valid_images, v_lines_all, h_lines_all, pts_all, H_all

    def get_reproj_error(self, K, R_all, t_all, world_pts, img_points):
        pts_reproj_all = []
        for i in range(len(img_points)):
            r_t = np.concatenate((R_all[i][:, :2], t_all[i].reshape(3, 1)), axis=1)
            pts = K @ r_t @ np.array(world_pts).T
            pts = pts / pts[-1, :]
            pts = pts[:2, :].T
            pts_reproj_all.append(pts)
            diff = np.abs(pts - img_points[i])
        return pts_reproj_all, np.mean(diff), np.var(diff), np.max(diff)

    def get_v_rows(self, h):
        h11, h12, h13 = h[:, 0]
        h21, h22, h23 = h[:, 1]
        v11 = np.array(
            [
                h11 * h11,
                h11 * h12 + h12 * h11,
                h12 * h12,
                h13 * h11 + h11 * h13,
                h13 * h12 + h12 * h13,
                h13 * h13,
            ]
        ).reshape(
            6,
        )

        v12 = np.array(
            [
                h11 * h21,
                h11 * h22 + h12 * h21,
                h12 * h22,
                h13 * h21 + h11 * h23,
                h13 * h22 + h12 * h23,
                h13 * h23,
            ]
        ).reshape(
            6,
        )

        v22 = np.array(
            [
                h21 * h21,
                h21 * h22 + h22 * h21,
                h22 * h22,
                h23 * h21 + h21 * h23,
                h23 * h22 + h22 * h23,
                h23 * h23,
            ]
        ).reshape(
            6,
        )

        V = np.array([v12, v11 - v22])
        return V

    def get_omega(self, H):
        V = []
        for h in H:
            V.append(self.get_v_rows(h))
        V = np.concatenate(V, axis=0)
        _, _, v_svd = np.linalg.svd(V)
        # pick the eigen vector corresponding to the smallest eigen value
        b = v_svd[-1]

        return np.array([[b[0], b[1], b[3]], [b[1], b[2], b[4]], [b[3], b[4], b[5]]])

    def get_K(self, w):
        """
        w: omega
        """
        y_0 = (w[0][1] * w[0][2] - w[0][0] * w[1][2]) / (
            w[0][0] * w[1][1] - w[0][1] ** 2
        )
        l = w[2][2] - (
            (w[0][2] ** 2) + y_0 * (w[0][1] * w[0][2] - w[0][0] * w[1][2])
        ) / (w[0][0])
        a_x = np.sqrt(l / w[0][0])
        a_y = np.sqrt(l * w[0][0] / (w[0][0] * w[1][1] - w[0][1] ** 2))
        s = -w[0][1] * a_x**2 * a_y / l
        x_0 = (s * y_0 / a_y) - (w[0][2] * a_x**2 / l)

        return [[a_x, s, x_0], [0, a_y, y_0], [0, 0, 1]]

    def get_R_t(self, K, H):
        R_t_all = []
        for h in H:
            r1_r2_t = np.linalg.inv(K) @ h
            zeta = 1 / np.linalg.norm(r1_r2_t[:, 0])
            r1_r2_t *= zeta
            r3 = np.cross(r1_r2_t[:, 0], r1_r2_t[:, 1]).reshape(3, 1)

            # conditioning the rotation matrix
            Q = np.concatenate((r1_r2_t[:, :2], r3), axis=1)
            u, _, vh = np.linalg.svd(Q)
            R = u @ vh

            R_t_all.append([R, r1_r2_t[:, 2]])

        return R_t_all

    def get_W(self, R_t):
        W_all = []
        for r, _ in R_t:
            phi = np.arccos((np.trace(r)-1)/ 2)
            W = (phi / (2*np.sin(phi))) * np.array([r[1][2] - r[2][1], r[2][0] - r[0][2], r[0][1] - r[1][0]])
            W_all.append(W)
        return W_all


    def R_from_W(self, W):
        R_all = []
        for w in W:
            # print(w)
            w_mat = np.array([[0.0, -w[2], w[1]], 
                            [w[2], 0.0, -w[0]], 
                            [-w[1], w[0], 0.0]])
                            
            phi = np.linalg.norm(w_mat)
            # w_mat = np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])
            r = np.eye(3) + (np.sin(phi)/phi) * w_mat + ((1- np.cos(phi)) / phi**2) * (w_mat@w_mat)
            R_all.append(r) 
        return R_all

    def cost_f(self, params, pts_all, actual_pts_aug):
        K = np.array(
            [[params[0], params[1], params[2]], 
            [0, params[3], params[4]], 
            [0, 0, 1]]
        )
        R_all = self.R_from_W(params[5:5+3*len(pts_all)].reshape(-1, 3))
        t_all = params[5+3*len(pts_all):].reshape(-1, 3)
        # print(R_all)
        diff_all = []
        for i, r_t in enumerate(list(zip(R_all, t_all))):
            r, t = r_t
            r_t = np.concatenate((r[:, :2], t.reshape(3, 1)), axis=1)
            
            pts = K @ r_t @ np.array(actual_pts_aug).T
            pts = pts / pts[-1, :]
            pts = pts[:2, :].T

            diff = np.array(pts_all[i]) - pts
            diff_all.append(diff.flatten())

        diff_all = np.concatenate(diff_all, axis=0)
        return diff_all