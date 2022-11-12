import os
import cv2 as cv
import numpy as np
from sklearn.cluster import KMeans



class CornerDetection:
    def __init__(self, data_dir) -> None:

        self.images = []
        for i in range(1, len(os.listdir(data_dir))+1):
            img_name = f'Pic_{i}.jpg'
            img = cv.imread(os.path.join(data_dir, img_name), 0)
            self.images.append(img)

        

    def get_lines(self, img, thres1=300, thres2=300):
        a = cv.Canny(img, thres1, thres2)
        lines = cv.HoughLines(a.astype("uint8"), 1, np.pi / 180, 52, None, 0, 0)
        lines = lines.squeeze(1)

        h_lines = []
        v_lines = []
        for l in lines:
            if np.abs(l[1] - np.pi / 2) < np.pi / 4:
                h_lines.append(l)
            else:
                v_lines.append(l)
        return lines, h_lines, v_lines

    def get_sorted_lines(self, v_lines, h_lines):

        v_lines = np.array(v_lines)
        x_intercept = v_lines[:, 0] / np.cos(v_lines[:, 1])

        h_lines = np.array(h_lines)
        y_intercept = h_lines[:, 0] / np.sin(h_lines[:, 1])

        v_kmeans = KMeans(n_clusters=8, random_state=0)
        v_kmeans.fit(x_intercept.reshape(-1, 1))

        h_kmeans = KMeans(n_clusters=10, random_state=0)
        h_kmeans.fit(y_intercept.reshape(-1, 1))

        v_lines_clustered = []
        for i in range(8):
            # rho_v, theta_v = np.mean(v_lines[v_kmeans.labels_ == i], axis=0)
            v_lines_clustered.append(
                list(np.mean(v_lines[v_kmeans.labels_ == i], axis=0))
            )

        v_lines_sorted = sorted(v_lines_clustered, key=lambda x: np.abs(x[0] / np.cos(x[1])))

        h_lines_clustered = []
        for i in range(10):
            # rho_v, theta_v = np.mean(v_lines[v_kmeans.labels_ == i], axis=0)
            h_lines_clustered.append(
                list(np.mean(h_lines[h_kmeans.labels_ == i], axis=0))
            )

        h_lines_sorted = sorted(h_lines_clustered, key=lambda x: np.abs(x[0] / np.sin(x[1])))

        return v_lines_sorted, h_lines_sorted

    def get_intersection_points(self, v_lines_sorted, h_lines_sorted):
        pts = []
        for v in v_lines_sorted:
            rho_v, theta_v = v
            v_homo_rep = np.array([np.cos(theta_v), np.sin(theta_v), -rho_v])
            v_homo_rep = v_homo_rep / v_homo_rep[-1]
            # pts = []
            for h in h_lines_sorted:
                rho_h, theta_h = h
                h_homo_rep = np.array([np.cos(theta_h), np.sin(theta_h), -rho_h])
                # if i == 0:
                h_homo_rep = h_homo_rep /h_homo_rep[-1]
                pt = np.cross(v_homo_rep, h_homo_rep)
                
                if pt[-1] != 0:
                    pt = pt / pt[-1]
                
                    pt = list(pt[:2].astype("int"))
                    pts.append(pt)
                
        return pts