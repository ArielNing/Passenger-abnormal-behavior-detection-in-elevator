# 电梯轿厢内乘客异常行为检测

[TOC]



### （一）运动前景提取

##### 1. 动态背景建模 

- 高斯建模包括单高斯和混合高斯
- 结果示例

![动态建模](C:\Users\ranyu.ning\Desktop\动态建模.PNG)

- 代码 

```python
fgbg = cv2.createBackgroundSubtractorMOG2() 
# fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()  # 单高斯
f=cv2.imread(path)
# 高斯模型提取背景 
frame = cv2.GaussianBlur(f, (5, 5), 0)
fgmask = fgbg.apply(frame)
thresh = cv2.threshold(fgmask, 40, 255, cv2.THRESH_BINARY)[1]
# 形态学处理
element = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
thresh = cv2.dilate(thresh, element, iterations=2)
thresh = cv2.erode(thresh, element, iterations=2)

cv2.imshow("Foreground", thresh)
```

##### 2. 静态背景建模 

- 包括平均值建模、color建模
- 平均值建模结果示例

![平均值建模](C:\Users\ranyu.ning\Desktop\平均值建模.png)

- 代码

```python
# 对每帧：
_, frame = c.read()
if frame is None:
    break

if avg is None:
    print("starting background model...")
    avg = frame.copy().astype("float")
    continue
cv2.accumulateWeighted(frame,avg,0.001)
diff=cv2.absdiff(frame,cv2.convertScaleAbs(avg))
diff=cv2.cvtColor(diff,cv2.COLOR_RGB2GRAY)
diff = cv2.GaussianBlur(diff, (5, 5), 0)
diff=cv2.threshold(diff,30,255,cv2.THRESH_BINARY)[1]
# 中值滤波去除噪点
diff = cv2.medianBlur(diff, 3)

cv2.imshow('diff',diff)
```

------

### （二）基于前景连通域的人数统计

- 计算前景的连通域，分别对每个连通域进行人数统计再加和，人数统计使用阈值法
- 结果示例

![人数估计1](C:\Users\ranyu.ning\Desktop\人数估计1.png)

![人数估计2](C:\Users\ranyu.ning\Desktop\人数估计2.png)

- 代码

```python
NP = 200 
minNP = 0.8 * NP
midNP = 1.8 * NP
mid2NP= 3.6 * NP
# find contours
(_, cnts, _) = cv2.findContours(diff.copy(), cv2.RETR_EXTERNAL,                     cv2.CHAIN_APPROX_SIMPLE)
for o in cnts:
    num = cv2.contourArea(o)
    if num < minNP:
        continue
    else:
        if num <= midNP:
            count += 1
            (x, y, w_o, h_o) = cv2.boundingRect(o)
            cv2.rectangle(f, (x, y), (x + w_o, y + h_o), (0, 255, 0), 2) # 绿色框 单人
            cv2.putText(f, 'single', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
        else:
            if num > mid2NP:
                mul = True
                (x, y, w_o, h_o) = cv2.boundingRect(o)
                cv2.rectangle(f, (x, y), (x + w_o, y + h_o), (255, 0, 0), 2) # 蓝色框 多人
                cv2.putText(f, 'multiple', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3,   (255, 0, 0), 1)
            else:
                count+=2
                (x, y, w_o, h_o) = cv2.boundingRect(o)
                cv2.rectangle(f, (x, y), (x + w_o, y + h_o), (0, 255, 255), 2) # 黄色框 两人 
                cv2.putText(f, 'two', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
```

------

### （三）多人异常行为检测

##### 1. 构造运动历史图像，初始化

- 运动历史图是经过计算得到的一副灰度图像，用来表征一段时间内同一位置像素点的运动情况。
- 代码

```python
# 运动历史图像初始化
motion_his = np.zeros((h, w), np.float32)
timestamp=0
# 参数初始化
MHI_DURATION = 10
```

##### 2. 每帧执行：更新运动历史图像，更新函数由帧差法定义 

- 代码

```python
# 帧差法
f_diff = cv2.absdiff(frame, prev_f)
f_diff = cv2.cvtColor(f_diff, cv2.COLOR_BGR2GRAY)
f_diff = cv2.threshold(f_diff, 60, 1, cv2.THRESH_BINARY)[1]
# 更新运动历史图像
timestamp+=1
cv2.motempl.updateMotionHistory(f_diff, motion_his, timestamp, MHI_DURATION)
mhi = np.uint8(np.clip((motion_his - (timestamp - MHI_DURATION)) / MHI_DURATION, 0, 1) * 255)
# 中值滤波去除噪点
mhi = cv2.medianBlur(mhi, 3)

cv2.imshow('mhi', mhi)
```

- 运动历史图像效果

![mhi](C:\Users\ranyu.ning\Desktop\mhi.png)

##### 3. 构造运动能量图，特征后处理，判别规则

- 运动能量图是由运动历史图计算得到的二值图像，用来表示运动的轮廓和能量的空间分布情况
- 特征后处理：运动能量图分别向水平方向和竖直方向投影，并计算水平方向和竖直方向的投影直方图，获得能量分布函数值E，建立判别规则
- 统计数据集中正常状态下能量分布函数值得分布区间B=[b1,b2]。如果E>>b2，则判定发生群体慌乱行为
- 代码

```python
# 设置投影直方图BIN
BIN_W=15
BIN_H=14

# 运动能量图
mei = cv2.threshold(mhi, 1, 1, cv2.THRESH_BINARY)[1]

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
    cv2.putText(f, 'Anomaly Detected', (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
(0, 0, 255), 2)
```

##### 4. 结果示例

![多人异常](C:\Users\ranyu.ning\Desktop\多人异常.png)

------

### （四）两人暴力行为检测

##### 1. 运动历史图

- 同上

##### 2. 特征提取 

- 计算能量函数，是在图像信息熵定义的基础上加了一个权重，即亮度值越大的像素点分配越高的权重
- 代码

```python
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
discriminate(res)
```

##### 3. 判别规则

- 对于能量值超过正常状态能量函数的统计值，判定发生了异常行为
- 代码

```python
def discriminate(res):
    u=14.3 # 正常状态下能量函数的平均值
    d=2.9 # 正常状态下能量函数值的标准差
    L=2 # 松弛因子
    if abs((res-u)/d)<=2:
        return False # 正常情况
    else:
        return False # 异常情况
```

##### 4. 结果示例

![双人异常](C:\Users\ranyu.ning\Desktop\双人异常.png)

------

### （五）单人异常行为检测

##### 1. 特征提取

- 提取人体目标，进行水平方向和竖直方向投影，提取粗略特征

##### 2. 特征后处理

- 投影子窗口法，在水平方向投影和竖直方向投影图像上取感兴趣区域所得窗口为检测子窗口，提取投影之后人体轮廓的长度和宽度信息，选取比例和人体轮廓的宽高比有关系。为消除人体不同身高对算法的影响，投影子窗口选取比例一般略微小于人体轮廓宽度和高度之比
- 水平方向投影子窗是在水平方向投影窗口的下侧起取窗口的五分之一的大小
- 竖直方向投影子窗口是在竖直方向投影窗口的左侧起取窗口的五分之一的大小

##### 3. 判别规则

- 分别计算投影子窗中非零像素点个数在该子窗中的占比Px、Py
- 得到视频帧的高宽H、W
- 代码

```python
# 定义规则
RATIO=0.2
STATE_S=lambda h,w:2*w/h # 站立
STATE_F=lambda h,w:0.5*w/h # 摔倒

h, w = f.shape[:2]
# 初始化
stand = STATE_S(h, w)
fail=STATE_F(h,w)
single_len_w=h*RATIO
single_len_h=w*RATIO
single_div_w=single_len_w*w
single_div_h=single_len_h*h
```

```python
# 对每帧：
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
```

------

### （六）遗留物检测

##### 1. 遗留物定义

- 轿厢内暂时静止的物体

##### 2. 特征提取

- 对每一帧都做一次帧差法和背景差法处理。

- 设FS是经过帧差法运算后的结果，FL是经过减背景运算后的结果

  | 序号 | FS(i,j) | FL(i,j) | 预估类型     |
  | ---- | ------- | ------- | ------------ |
  | 1    | 1       | 1       | 运动对象     |
  | 2    | 1       | 0       | 噪声         |
  | 3    | 0       | 1       | 暂时静止对象 |
  | 4    | 0       | 0       | 纯背景       |

##### 3. 构造统计信息可能性函数H

##### 4. 判别规则 H(i,j)>thrs_H

```python
# 初始化
minNL=50
img_H=np.zeros((h,w),np.uint8)
thrs_H=30  # 遗留物检测阈值
```

```python
# 初始化
minNL=50
img_H=np.zeros((h,w),np.uint8)
thrs_H=30  # 遗留物检测阈值

# 对每帧：
# 统计信息可能性函数
mask = cv2.bitwise_and(cv2.bitwise_not(f_diff),diff)
img_H = cv2.add(img_H,cv2.bitwise_and(mask,img_H))
img_H = cv2.threshold(img_H, thrs_H, 255, cv2.THRESH_BINARY)[1]
(_, lcnts, _) = cv2.findContours(img_H.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for o in lcnts:
    num = cv2.contourArea(o)
    if num < minNL:
        continue
    (x, y, w_o, h_o) = cv2.boundingRect(o)
    cv2.rectangle(f, (x, y), (x + w_o, y + h_o), (0, 0, 255), 2) # 红色框
```

