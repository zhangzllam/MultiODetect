import os.path

import cv2
import numpy as np
import math
import pandas as pd
# 1 2 5 10 15 20 30 40 50 60

open= 3
db = [10]
speed_set=[10]
time = [200]
num = 20
folder_path = r'.\rotating_multi_frames'
res_dir = r'.\res'
caled_dic = {}
for et in time:
    for voa in db:
        lst = []
        for speed in speed_set:
            start = 2
            add= 1
            ref=speed*et/1000*(add)
            caled_speed = []
            for i in range(num-start-add+1-1):

                frame1 = cv2.imread( folder_path + '\\'+ str(open)+'\\'+str(voa) +'\\'+str(et)+'\\'+str(speed)+"\\angle_"+str(open)+'_'+str(voa)+'db_'+str(et)+'ms_'+str(speed)+'hz_'+str(start+i)+'.png', cv2.IMREAD_GRAYSCALE)


                frame2 = cv2.imread( folder_path +'\\' + str(open)+'\\'+str(voa) +'\\'+str(et)+'\\'+str(speed)+"\\angle_"+str(open)+'_'+str(voa)+'db_'+str(et)+'ms_'+str(speed)+'hz_'+str(add+start+i)+'.png', cv2.IMREAD_GRAYSCALE)

                feature_params = dict( maxCorners = 300,
                                       qualityLevel = 0.7,
                                       minDistance = 20)

                lk_params = dict( winSize  = (30,30),
                                  maxLevel = 3)

                color = np.random.randint(0,255,(300,3))


                p0 = cv2.goodFeaturesToTrack(frame1, mask = None, **feature_params)

                mask = np.zeros_like(frame1)

                p1, st, err = cv2.calcOpticalFlowPyrLK(frame1, frame2, p0, None, **lk_params)

                good_new = p1[st == 1]
                good_old = p0[st == 1]

                M, _ = cv2.estimateAffine2D(good_old, good_new)


                try:
                    rotation_angle = np.arctan2(M[1, 0], M[0, 0])
                    rotation_angle = rotation_angle * (180.0 / math.pi)
                except TypeError:
                    rotation_angle = 0


                cul_speed = rotation_angle/(add*et/1000)
                caled_speed.append(cul_speed)

                '''for i, (new, old) in enumerate(zip(good_new, good_old)):
                    a, b = new.ravel()
                    c, d = old.ravel()
                    mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
                    frame = cv2.circle(frame1, (int(a), int(b)), 5, color[i].tolist(), -1)
                    img = cv2.add(frame1, mask)
                cv2.imshow('frame', img)
                cv2.waitKey()
                cv2.destroyAllWindows()'''

            print(f'Opening {open},'
                  f' {speed}Hz,'
                  f' {et}ms,'
                  f'ADD {add} ')
            print(f'Calculated & Estimate  {speed:.4f}', f'Cal_speed {sum(caled_speed) / len(caled_speed):.4f}')
            lst.append(sum(caled_speed) / len(caled_speed))

        caled_dic[str(et)] = lst







