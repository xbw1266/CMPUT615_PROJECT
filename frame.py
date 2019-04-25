import numpy as np
from toolbox import extraction
from toolbox import normalize


# The frame class has a few parameters:
# raw_pts are points directly extract from the frame
# key_pts are key points that have a match in the adjacent frame
class Frame(object):
    def __init__(self, mapp, K, Kinv, img, pose=np.eye(4), tid=None):
        self.K = np.array(K)
        self.pose = np.array(pose)
        self.Kinv = Kinv

        if img is None:
            self.h, self.w = 0, 0
            self.raw_pts, self.des, self.key_pts = None, None, None
        else:
            self.h, self.w = img.shape[0:2]
            self.raw_pts, self.des = extraction(img)
            self.key_pts = [None]*len(self.raw_pts)
        self.id = tid if tid is not None else mapp.add_frame(self)

# using property to boost the performance:
    @property
    def nps(self):
        if not hasattr(self, '_nps'):
            self._nps = normalize(self.Kinv, self.raw_pts)
        return self._nps

