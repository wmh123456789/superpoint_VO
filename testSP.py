import numpy as np
import cv2

from sp_extractor import SuperPointFrontend, PointTracker

def testSingleFrame(imgfile):
    detector = SuperPointFrontend(weights_path="superpoint_v1.pth",
                                  nms_dist=4,
                                  conf_thresh=0.015,
                                  nn_thresh=0.7,
                                  cuda=False)
    frame = cv2.imread(imgfile)
    pts, desc, heatmap = detector.run(frame)
    print("Find {} superpoints".format(len(pts)))

pass



def testMatching(imgfile0, imgfile1):
    detector = SuperPointFrontend(weights_path="superpoint_v1.pth",
                                  nms_dist=4,
                                  conf_thresh=0.015,
                                  nn_thresh=0.7,
                                  cuda=False)
    frame0 = cv2.imread(imgfile0)
    frame1 = cv2.imread(imgfile1)
    pts0, desc0, heat0 = detector.run(frame0)
    pts1, desc1, heat1 = detector.run(frame1)


    pass



if __name__ == '__main__':
    img0 = 'img/form0.jpg'
    testSingleFrame()
    testMatching()