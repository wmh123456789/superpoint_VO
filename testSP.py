import os
import numpy as np
import cv2

from sp_extractor import SuperPointFrontend, PointTracker

def testSingleFrame(imgfile,rstImage='rst_SuperPoint.jpg'):
    detector = SuperPointFrontend(weights_path="superpoint_v1.pth",
                                  nms_dist=4,
                                  conf_thresh=0.015,
                                  nn_thresh=0.7,
                                  cuda=False)
    frame = cv2.imread(imgfile)
    pts, desc, heatmap = detector.run(frame)
    pt_cnt = 0
    for x,y,k in zip(pts[0],pts[1],pts[2]):
        if k > 0.2:
            pt_cnt += 1
            cv2.circle(frame,(int(x),int(y)),5,(0,0,255),2)
    cv2.imwrite(rstImage, frame)
    # cv2.imshow('SuperPoint', frame)
    # cv2.waitKey(0)
    print("In {}, find {} superpoints, {} are circled".format(imgfile,len(pts[0]),pt_cnt))

pass

def testFolder(imgRoot):
    files = os.listdir(imgRoot)
    for imgFile in files:
        rst_file = os.path.join(imgRoot, "sp_"+imgFile)
        testSingleFrame(os.path.join(imgRoot,imgFile),rst_file)



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
    print ('find {} in {}, and find {} in {}'.format(len(pts0[0]),imgfile0,len(pts1[0]),imgfile1))



    pass



if __name__ == '__main__':
    img0 = 'img/form0.jpg'
    img1 = 'img/form1.jpg'
    # testSingleFrame(img0)
    # testMatching()

    imgroot = 'img/shot/'
    testFolder(imgroot)