import glob
import math

import cv2
import matplotlib.pyplot as plt
import numpy as np


def get_calibration_matrix(path):
    """Returns camera calibration matrix using
    chessboard images in given glob path
    assuming 6x9 or 5x9 chess corners
    """
    object_points = []  # 3d point in real world space
    img_points = []  # 2d points in image plane.

    images = glob.glob(path)
    total_image_count = len(images)

    image_count = 1
    fig = plt.figure()
    for filename in images:
        img = cv2.imread(filename)
        nx, ny = 6, 9
        retval, corners = cv2.findChessboardCorners(img, (nx, ny))
        objp = np.zeros((nx * ny, 3), np.float32)
        objp[:, :2] = np.mgrid[0: nx, 0: ny].T.reshape(-1, 2)

        if not retval:
            nx, ny = 5, 9  # Trying with 5 rows
            objp = np.zeros((nx * ny, 3), np.float32)
            objp[:, :2] = np.mgrid[0: nx, 0: ny].T.reshape(-1, 2)
            retval, corners = cv2.findChessboardCorners(img, (nx, ny))

        if retval:
            object_points.append(objp)
            img_points.append(corners)

            ax = fig.add_subplot(math.ceil(total_image_count / 2), 2, image_count)
            chessboard_with_corners = cv2.drawChessboardCorners(img, (nx, ny), corners, retval)
            chessboard_with_corners = cv2.cvtColor(chessboard_with_corners, cv2.COLOR_BGR2RGB)
            ax.imshow(chessboard_with_corners)
            ax.axis('off')
            image_count += 1

    return cv2.calibrateCamera(object_points, img_points, img.shape[0:2], None, None), fig


def undistort(img, cameraMatrix, distCoeffs):
    """Returns undistorted image using given
    object points and image points
    """
    return cv2.undistort(img, cameraMatrix, distCoeffs)
