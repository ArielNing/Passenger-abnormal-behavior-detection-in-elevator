import cv2
import numpy as np
import math

MHI_DURATION = 10
DEFAULT_THRESHOLD = 32
MAX_TIME_DELTA = 0.25
MIN_TIME_DELTA = 0.05
BIN_W=15
BIN_H=14
RATIO=0.2
STATE_S=lambda h,w:2*w/h
STATE_F=lambda h,w:0.5*w/h
NP = 200
minNP = 0.8 * NP
midNP = 1.8 * NP
mid2NP= 3.6 * NP
minNL=50

def discriminate(res):
    u=14.3
    d=2.9
    L=2
    if abs((res-u)/d)<=2:
        return False #正常情况
    else:
        return False #异常情况

def videoDetector(video_path):
    c = cv2.VideoCapture(video_path) 

    # 保存视频
    # fourcc=cv2.VideoWriter_fourcc(*'XVID')
    # out_img=cv2.VideoWriter('output/normal.avi',fourcc,20.0,(400,300))
    # out_mhi=cv2.VideoWriter('output/normal_mhi.avi',fourcc,20,(400,300),False)

    _,f=c.read()
    f=cv2.resize(f,(400,300))
    prev_f = f.copy()
    avg = None


    h, w = f.shape[:2]
    # 初始化
    stand = STATE_S(h, w)
    fail=STATE_F(h,w)
    single_len_w=h*RATIO
    single_len_h=w*RATIO
    single_div_w=single_len_w*w
    single_div_h=single_len_h*h

    # 运动历史图像初始化
    motion_his = np.zeros((h, w), np.float32)
    img_H=np.zeros((h,w),np.uint8)
    timestamp=0


    while (1):
        count = 0 # 人数
        mul = False
        _, frame = c.read()
        if frame is None:
            break
        frame = cv2.resize(frame, (400, 300))

        timestamp+=1
        # f用于绘制展示
        f=frame.copy()
        # 帧差法
        f_diff = cv2.absdiff(frame, prev_f)
        f_diff = cv2.cvtColor(f_diff, cv2.COLOR_BGR2GRAY)
        f_diff = cv2.threshold(f_diff, 60, 1, cv2.THRESH_BINARY)[1]


        if avg is None:
            print("starting background model...")
            avg = frame.copy().astype("float")
            continue
        cv2.accumulateWeighted(frame,avg,0.001) # 0.001
        diff=cv2.absdiff(frame,cv2.convertScaleAbs(avg))

        diff=cv2.cvtColor(diff,cv2.COLOR_RGB2GRAY)
        diff = cv2.GaussianBlur(diff, (5, 5), 0)
        diff=cv2.threshold(diff,30,1,cv2.THRESH_BINARY)[1]
        # 中值滤波去除噪点
        diff = cv2.medianBlur(diff, 3)


        #统计信息可能性函数
        mask=cv2.bitwise_and(cv2.bitwise_not(f_diff),diff)
        img_H=cv2.add(img_H,cv2.bitwise_and(mask,img_H))
        thrs_H=30 # 遗留物检测阈值
        img_H = cv2.threshold(img_H, thrs_H, 255, cv2.THRESH_BINARY)[1]
        # cv2.imshow('遗留物检测-H',img_H)
        (_, lcnts, _)=cv2.findContours(img_H.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for o in lcnts:
            num=cv2.contourArea(o)
            if num<minNL:
                continue
            (x, y, w_o, h_o) = cv2.boundingRect(o)
            cv2.rectangle(f, (x, y), (x + w_o, y + h_o), (0, 0, 255), 2)  # 红色框

        # 运动历史图像
        cv2.motempl.updateMotionHistory(f_diff, motion_his, timestamp, MHI_DURATION)
        mhi = np.uint8(np.clip((motion_his - (timestamp - MHI_DURATION)) / MHI_DURATION, 0, 1) * 255)
        # 中值滤波去除噪点
        mhi = cv2.medianBlur(mhi, 3)
        cv2.imshow('mhi', mhi)


        # 判断人数
        # cv2.imshow('diff', img_diff)
        # find contours
        (_, cnts, _) = cv2.findContours(diff.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for o in cnts:
            num = cv2.contourArea(o)
            if num < minNP:
                continue
            else:
                if num <= midNP:
                    count += 1
                    (x, y, w_o, h_o) = cv2.boundingRect(o)
                    cv2.rectangle(f, (x, y), (x + w_o, y + h_o), (0, 255, 0), 2)  # 绿色框 单人
                    cv2.putText(f, 'single', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
                else:
                    if num > mid2NP:
                        mul = True
                        (x, y, w_o, h_o) = cv2.boundingRect(o)
                        cv2.rectangle(f, (x, y), (x + w_o, y + h_o), (255, 0, 0), 2)  # 蓝色框 多人 255 0 0
                        cv2.putText(f, 'multiple', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
                    else:
                        count+=2
                        (x, y, w_o, h_o) = cv2.boundingRect(o)
                        cv2.rectangle(f, (x, y), (x + w_o, y + h_o), (0, 255, 255), 2)  # 黄色框 两人 0 255 255
                        cv2.putText(f, 'two', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
        if count==0:
            continue
        if count==1 and mul is False:
            print('单人-----------------------')
            cv2.putText(f,'single person',(10,10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            # 前景提取水平投影
            num_w = [0 for z in range(0, w)]
            for i in range(w):
                for j in range(h):
                    if diff[j][i] == 1:
                        num_w[i] += 1
            sum_w=sum(map(lambda x:x if x<=single_len_w else single_len_w,num_w))
            p_w=sum_w/single_div_w
            # 前景提取垂直投影
            num_h = [0 for z in range(0, h)]
            for i in range(h):
                for j in range(w):
                    if diff[i][j] == 1:
                        num_h[i] += 1
            sum_h = sum(map(lambda x: x if x <= single_len_h else single_len_h, num_h))
            p_h = sum_h / single_div_h
            E=p_h/p_w
            if E>=stand:
                print('stand')
            else:
                if E<=fail:
                    print('failed down')
                else:
                    print('bow')
        else:

            # 运动历史图像
            cv2.motempl.updateMotionHistory(f_diff, motion_his, timestamp, MHI_DURATION)
            mhi = np.uint8(np.clip((motion_his - (timestamp - MHI_DURATION)) / MHI_DURATION, 0, 1) * 255)
            # 中值滤波去除噪点
            mhi = cv2.medianBlur(mhi, 3)

            cv2.imshow('mhi', mhi)

            if mul or count>2:
                print('多人-----------------------')
                cv2.putText(f, 'multiple people', (10,10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                # 运动能量图
                mei = cv2.threshold(mhi, 1, 1, cv2.THRESH_BINARY)[1]
                # cv2.imshow('mei',mei)

                # 计算水平投影直方图
                hist_w = [0 for z in range(0, BIN_W)]
                div_w = [0 for z in range(0, BIN_W)]
                len_w = w / BIN_W
                if len_w > int(len_w):
                    len_w = int(len_w) + 1
                else:
                    len_w = int(len_w)
                for i in range(w):
                    index = int(i / len_w)
                    div_w[index] += h
                    for j in range(h):
                        if mei[j][i] == 1:
                            hist_w[index] += 1

                p_w = np.sum([h / d for h, d in zip(hist_w, div_w)])
                print("p_w:" + str(p_w))

                # 计算垂直投影直方图
                hist_h = [0 for z in range(0, BIN_H)]
                div_h = [0 for z in range(0, BIN_H)]
                len_h = h / BIN_H
                if len_h > int(len_h):
                    len_h = int(len_h) + 1
                else:
                    len_h = int(len_h)
                for i in range(h):
                    index = int(i / len_h)
                    div_h[index] += w
                    for j in range(w):
                        if mei[i][j] == 1:
                            hist_h[index] += 1
                p_h = np.sum([h / d for h, d in zip(hist_h, div_h)])
                print("p_h:" + str(p_h))

                E = p_w + p_h
                print("total:" + str(E))

                cv2.putText(f, 'Panic Index:' + str(round(E,2)), (150, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 0, 255), 1)

                if E>=0.5:
                    print("多人异常")
                    cv2.putText(f, 'Anomaly Detected',  (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (0, 0, 255), 2)
                    

            else:
                print('两人-----------------------')
                cv2.putText(f, 'multiple people', (10,10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                # 能量函数计算
                tmp = []
                for i in range(256):
                    tmp.append(0)
                k = 0
                res = 0
                for i in range(h):
                    for j in range(w):
                        val = int(mhi[i][j])
                        tmp[val] = float(tmp[val] + 1)
                        k = float(k + 1)
                for i in range(len(tmp)):
                    tmp[i] = float(tmp[i] / k)
                for i in range(len(tmp)):
                    if (tmp[i] == 0):
                        continue
                    else:
                        res = float(res - (i / 255.0) * tmp[i] * (math.log(tmp[i]) / math.log(2.0)))
                print(res)
                cv2.putText(f, 'Violence Index:' + str(round(res, 2)), (150, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 0, 255), 1)
                # discriminate(res)

        prev_f=frame.copy() # 传入prev_f
        cv2.imshow('img', f)
        # out_img.write(f)
        # out_mhi.write(mhi)

        if 0xFF&cv2.waitKey(5)==27:
            break

    cv2.destroyAllWindows()
    c.release()
    # out_img.release()
    # out_mhi.release()

