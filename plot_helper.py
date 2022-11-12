from copy import deepcopy
import cv2 as cv
import numpy as np


def draw_lines(img, lines, c=(0, 0, 255)):
    cdst = deepcopy(img)
    for l in lines:
        rho = l[0]
        theta = l[1]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
        pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
        cv.line(cdst, pt1, pt2, c, 2)
    return cdst

def draw_points(pts, image, path="solutions"):
    img = deepcopy(image)
    for pt in pts:
        img = cv.circle(img, pt, 6, (0, 0, 255), -1)

    cv.imwrite(f"{path}/points.jpg", img)

def draw_reproj_points(data_dir, img_list, pts_all, pts_reproj_all, plot_path="solutions/reproj"):
    color_img = []
    for i in img_list:
        img_name = f'{data_dir}/Pic_{i+1}.jpg'
        img = cv.imread(img_name)
        color_img.append(img)
    for j, pts in enumerate(zip(pts_all, pts_reproj_all)):
        pts_orig, pts_rproj = pts
        img = deepcopy(color_img[j])
        for i in range(80):
            # print(f"pt: {tuple(pt.astype('int'))}")
            pt = tuple(pts_rproj[i].astype('int'))
            cv.circle(img, pt, 2, (0, 0, 255), -1)


            pt_0 = tuple(pts_orig[i])
            cv.circle(img, pt_0, 2, (0, 255, 0), -1)
            cv.putText(
                img,
                str(i),
                (pt_0[0] - 10, pt_0[1] + 10),
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                1,
            )
        cv.imwrite(f"{plot_path}_{img_list[j]+1}.jpg", img)
