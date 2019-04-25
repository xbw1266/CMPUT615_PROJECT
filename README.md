A python implementation of ORB SLAM for CMPUT615 course project:  
Library used:  
numpy - for svd and matrix inverse etc.  
skimage - for RANSAC  
cv2 - image handling and ORB extraction  
g2o - bundle adjustment / optimizer  
pangolin - map viewer  
sdl2 - video/image sequence player  
  
How to use:  
1. Change camera paramerters (f, cx, cy (cx = W//2, cy=H//2 by default but you can change them if they are not))  
2. Change the flag in the slam.py: flag == 1 for video sequence, flag == 2 for image sequene  
3. Modify the path to the directory of the image sequence or video file if you want to test other data  
4. KITTI image sequence is removed from github due to the upload size limit, you can redownload them from kitti website. Otherwise, slam.py will return nothing because the original kitti image sequences in my computer is not in the repository.  
