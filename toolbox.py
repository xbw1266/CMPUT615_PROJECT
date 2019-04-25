import cv2
import numpy as np
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform
from skimage.transform import EssentialMatrixTransform

# change numpy setting to suppress the scientific notation:
np.set_printoptions(suppress=True)


# change to homogeneous coordinates:
def homogeneous(x):
    return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)


def pose_homogeneous(r, t):
    ret = np.eye(4)
    ret[:3, :3] = r
    ret[:3, 3] = t
    return ret


# get the pose of the current frame from the Essential or Fundamental matrix:
def get_pose(F):
    # Using fundamental matrix
    # http://answers.opencv.org/question/206817/extract-rotation-and-translation-from-fundamental-matrix/
    # there are exactly two pairs (R,T) corresponding to each essential matrix E
    #  https://en.wikipedia.org/wiki/Essential_matrix#Determining_R_and_t_from_E
    W = np.mat([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    U, d, Vt = np.linalg.svd(F)
    if np.linalg.det(U) < 0:
        U *= -1.0
    if np.linalg.det(Vt) < 0:
        Vt *= -1.0
    # print('U dim:', np.size(U))
    # print('W dim:', np.size(W))
    R = np.dot(np.dot(U, W), Vt)
    if np.sum(R.diagonal()) < 0:
        R = np.dot(np.dot(U, W.T), Vt)
    t = U[:, 2]

    if t[2] < 0:
        t *= -1

    pose = np.linalg.inv(pose_homogeneous(R, t))
    # print(pose)
    return pose


# Extraction:
# input raw image and the number of feature points want to track

#
def extraction(img):
    # using ORB extractor as the main extractor:
    # initialize:
    orb = cv2.ORB_create()

    # detection:
    # (alternative option can be harris corner etc. )
    pts = cv2.goodFeaturesToTrack(np.mean(img, axis=2).astype(np.uint8), 3000, qualityLevel=0.01, minDistance=7)

    kps = [cv2.KeyPoint(x=f[0][0], y=f[0][1], _size=20) for f in pts]
    # extraction
    kps, des = orb.compute(img, kps)
    return np.array([(kp.pt[0], kp.pt[1]) for kp in kps]), des
    # TODO: detect good features first


# Match between two frames (frames are in Frame class):
def match_frame(f1, f2):
    # using cv2 BFMatcher:
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(f1.des, f2.des, k=2)

    # apply Lowe's ratio test (ref: OpenCv document):
    # we want both the coordinates and the indices of the points
    ret = []
    idx1 = []
    idx2 = []

    # TODO: remove the identical points:
    idx1s, idx2s = set(), set()
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            p1 = f1.nps[m.queryIdx]
            p2 = f2.nps[m.trainIdx]
            if m.distance < 32:
                if m.queryIdx not in idx1s and m.trainIdx not in idx2s:
                    idx1.append(m.queryIdx)
                    idx2.append(m.trainIdx)
                    idx1s.add(m.queryIdx)
                    idx2s.add(m.trainIdx)
                    ret.append((p1, p2))

    # remove duplicate points
    assert (len(set(idx1)) == len(idx1))
    assert (len(set(idx2)) == len(idx2))

    assert len(ret) >= 8
    ret = np.array(ret)
    idx1 = np.array(idx1)
    idx2 = np.array(idx2)

    # filter:
    model, inliers = ransac((ret[:, 0], ret[:, 1]),
                            FundamentalMatrixTransform,
                            min_samples=8,
                            residual_threshold=0.001,
                            max_trials=100)

    # model.params will return the essential matrix or the fundamental matrix
    Rt = get_pose(model.params)
    print('Matches: %d' % (sum(inliers)))

    return idx1[inliers], idx2[inliers], Rt, model.params


# normalize the points:
def normalize(kinv, pts):
    return np.dot(kinv, homogeneous(pts).T).T[:, 0:2]


def denormalize(K, pts):
    ret = np.dot(K, np.array([pts[0], pts[1], 1.0]))
    ret /= ret[2]
    return int(round(ret[0])), int(round(ret[1]))


# Using DLT: triangulation to calculate the points in 3D coordinates
# ref: linear triangulation method from ORB_SLAM2
def triangulate(frame1, frame2, pts1, pts2):
    # using the poses and the key points in each frame to calculate the 3D coordinates of the key points
    pose1 = frame1.pose
    pose2 = frame2.pose
    ret = np.zeros((pts1.shape[0], 4))
    for i, p in enumerate(zip(pts1, pts2)):
        A = np.zeros((4, 4))
        A[0] = p[0][0] * pose1[2] - pose1[0]
        A[1] = p[0][1] * pose1[2] - pose1[1]
        A[2] = p[1][0] * pose2[2] - pose2[0]
        A[3] = p[1][1] * pose2[2] - pose2[1]
        _, _, vt = np.linalg.svd(A)
        ret[i] = vt[3]
    return ret


def hamming_distance(a, b):
    r = (1 << np.arange(8))[:,None]
    return np.count_nonzero((np.bitwise_xor(a, b) & r) != 0)

