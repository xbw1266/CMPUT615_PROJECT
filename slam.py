import cv2
from visualize import Mapping, Visualize
from frame import Frame
from point import Point, Map
import numpy as np
from process_frame import process_frame
import os

# Intrinsic parameters:
F = 984
# assuming no scaling factor
# assuming principal point at center of the frame

# reading video sequence:
cap = cv2.VideoCapture("test_kitti984.mp4")
path = os.getcwd()+"/" + "kitti/"

    
# if the test is image sequence (e.g. KITTI database):
flag = 2

# using flag to run video or image sequence:\


if flag == 1:
    # live video test
    # cap = cv2.VideoCapture(-1)
    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if i == 0:
            print("Loading successful")
            # determine intrinsic parameters:
            W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            # since the mapping used width 1024:
            if W > 1024:
                downscale = 1024.0/W

                # doing this to keep the aspect ratio of the original image
                F *= downscale
                H = int(H * downscale)
                W = 1024
            K = np.array([[F, 0, W // 2], [0, F, H // 2], [0, 0, 1]])
            Kinv = np.linalg.inv(K)

            # initialize:
            mapp = Map()
            plot2d = Visualize(W, H)
            plot3d = Mapping()
            i += 1
            continue
        if ret is True:
            img = cv2.resize(frame, (W, H))
            process_frame(mapp, img, K, Kinv, plot2d, plot3d)
        else:
            break
        i += 1

else:
    files = os.listdir(path)
    files.sort()
    for i in range(len(files)):
        pathfull = path + files[i]
        frame = cv2.imread(pathfull)

        if i == 0:
            print("Loading successful")
            H, W, C = frame.shape
            if W > 1024:
                downscale = 1024.0/W

                # doing this to keep the aspect ratio of the original image
                F *= downscale
                H = int(H * downscale)
                W = 1024
            K = np.array([[F, 0, W // 2], [0, F, H // 2], [0, 0, 1]])
            Kinv = np.linalg.inv(K)

            # initialize:
            mapp = Map()
            plot2d = Visualize(W, H)
            plot3d = Mapping()
            continue

        img = cv2.resize(frame, (W, H))
        process_frame(mapp, img, K, Kinv, plot2d, plot3d)





