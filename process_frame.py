import cv2
import g2o
from visualize import Mapping, Visualize
from frame import Frame
from toolbox import match_frame, triangulate, homogeneous, normalize
from point import Point, Map
import numpy as np
import time


# assume the initial pose is eye
def process_frame(mapp, img, K, Kinv, plot2d, plot3d):
    t1 = time.time()
    # first resize:
    frame = Frame(mapp, K, Kinv, img)
    if frame.id == 0:
        return
    # current frame
    f1 = mapp.frames[-1]
    # previous frame
    f2 = mapp.frames[-2]
    print(f1.id)
    # once we have two frames:

    # step1: match two images to get point indices and Rt
    # these indices whose points correspond have passed RANSAC filter
    idx1, idx2, pose_matrix, _ = match_frame(f1, f2)
    f1.pose = np.dot(pose_matrix, f2.pose)

    # step2:
    # add new observation if the point is already observed in the previous frame
    for i, idx in enumerate(idx2):
        if f2.key_pts[idx] is not None and f1.key_pts[idx1[i]] is None:
            f2.key_pts[idx].add(f1, idx1[i])

    # idx1 and idx2 correspond to the same list of points but in two adjacent frames
    # because key points have the same length as raw points:
    # idx1 is the index of the points that have match for
    # we want to triangulate the key points that we don't have match for:
    good_pts = np.array([f1.key_pts[ii] is None for ii in idx1])

    # using normalized coordinates to compute triangulation
    # this will return the key points' homogeneous coordinates:
    pts_homo = triangulate(f1, f2, normalize(f1.Kinv, f1.raw_pts[idx1]), normalize(f2.Kinv, f2.raw_pts[idx2]))

    good_pts &= np.abs(pts_homo[:, 3]) != 0
    pts_homo /= pts_homo[:, 3:]

    # for a qualified point to be added to the map, it has to pass the following tests:
    new_pts_count = 0
    for i, p in enumerate(pts_homo):
        if not good_pts[i]:
            continue

        # check if the points are in front of the camera
        pl1 = np.dot(f1.pose, p)
        pl2 = np.dot(f2.pose, p)
        if pl1[2] < 0 or pl2[2] < 0:
            continue

        # reproject:
        pp1 = np.dot(K, pl1[:3])
        pp2 = np.dot(K, pl2[:3])

        # check reprojection error:
        pp1 = (pp1[0:2] / pp1[2]) - f1.raw_pts[idx1[i]]
        pp2 = (pp2[0:2] / pp2[2]) - f2.raw_pts[idx2[i]]
        pp1 = np.sum(pp1 ** 2)
        pp2 = np.sum(pp2 ** 2)
        if pp1 > 2 or pp2 > 2:
            continue

        # using the colored pixel in the frame at the point location to color the point
        color = img[int(round(f1.raw_pts[idx1[i], 1])), int(round(f1.raw_pts[idx1[i], 0]))]
        pt = Point(mapp, p[0:3], color)
        pt.add(f1, idx1[i])
        pt.add(f2, idx2[i])
        new_pts_count += 1
    # by these two filters, adding the qualified points
    print("New points added: %d" % new_pts_count)

    # now display the result (in both 2D and 3D mapping):
    if plot2d is not None:
        # paint annotations on the image
        for i1, i2 in zip(idx1, idx2):
            u1, v1 = int(round(f1.raw_pts[i1][0])), int(round(f1.raw_pts[i1][1]))
            u2, v2 = int(round(f2.raw_pts[i2][0])), int(round(f2.raw_pts[i2][1]))

            # this is for plotting the matching line:
            if f1.key_pts[i1] is not None:
                if len(f1.key_pts[i1].frames) >= 5:
                    cv2.circle(img, (u1, v1), color=(0, 255, 0), radius=3)
                else:
                    cv2.circle(img, (u1, v1), color=(0, 0, 255), radius=3)
            else:
                cv2.circle(img, (u1, v1), color=(0, 0, 0), radius=3)
            cv2.line(img, (u1, v1), (u2, v2), color=(255, 0, 0))
            # font = cv2.FONT_HERSHEY_SIMPLEX
            # cv2.putText(img, 'FPS: %.2F' % (1/(time.time()-t1)), (10, 500), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
        plot2d.draw(img)

    # Now doing Bundle Adjustment for every 10 frames:

    if frame.id >= 9 and frame.id % 10 == 0:
        err = mapp.optimize()
        print("Optimizing the frame: %f units of error" % err)

    # now display the 3D map:
    if plot3d is not None:
         plot3d.draw(mapp)

    print('FPS: %.1f' % (1/(time.time()-t1)))

