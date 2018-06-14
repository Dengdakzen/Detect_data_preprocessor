import numpy as np
from matplotlib import pyplot as plt
import cv2
import json

def fun1():  
    img = cv2.imread('../Detect_data/Images/0.jpg',cv2.IMREAD_GRAYSCALE)  
    #bins->图像中分为多少格；range->图像中数字范围  
    plt.hist(img.ravel(), bins=256, range=[0, 256])  
    plt.show()  
  
def fun2(img):  
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    color = ('b', 'g', 'r')
    for i, col in enumerate(color):
        histr = cv2.calcHist([hsv], [i], None, [256], [0, 256])  
        plt.plot(histr, color=col)  
    # plt.xlim([0, 256])  
    plt.show() 

def fun3(img1,img2):  
    hsv1 = cv2.cvtColor(img1,cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(img2,cv2.COLOR_BGR2HSV)
    h1 = cv2.calcHist([hsv1], [0], None, [256], [0, 256])
    h2 = cv2.calcHist([hsv2], [0], None, [256], [0, 256])
    s1 = cv2.calcHist([hsv1], [1], None, [256], [0, 180])
    s2 = cv2.calcHist([hsv2], [1], None, [256], [0, 180])
    # cv2.normalize()
    r1 = cv2.compareHist(h1,h2,cv2.HISTCMP_CORREL)
    r2 = cv2.compareHist(s1,s2,cv2.HISTCMP_CORREL)
    print(r1,r2)


    

def color_moments(filename):
    img = cv2.imread(filename)
    if img is None:
        return
    # Convert BGR to HSV colorspace
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Split the channels - h,s,v
    h, s, v = cv2.split(hsv)
    # Initialize the color feature
    color_feature = []
    # N = h.shape[0] * h.shape[1]
    # The first central moment - average 
    h_mean = np.mean(h)  # np.sum(h)/float(N)
    s_mean = np.mean(s)  # np.sum(s)/float(N)
    v_mean = np.mean(v)  # np.sum(v)/float(N)
    color_feature.extend([h_mean, s_mean, v_mean])
    # The second central moment - standard deviation
    h_std = np.std(h)  # np.sqrt(np.mean(abs(h - h.mean())**2))
    s_std = np.std(s)  # np.sqrt(np.mean(abs(s - s.mean())**2))
    v_std = np.std(v)  # np.sqrt(np.mean(abs(v - v.mean())**2))
    color_feature.extend([h_std, s_std, v_std])
    # The third central moment - the third root of the skewness
    h_skewness = np.mean(abs(h - h.mean())**3)
    s_skewness = np.mean(abs(s - s.mean())**3)
    v_skewness = np.mean(abs(v - v.mean())**3)
    h_thirdMoment = h_skewness**(1./3)
    s_thirdMoment = s_skewness**(1./3)
    v_thirdMoment = v_skewness**(1./3)
    color_feature.extend([h_thirdMoment, s_thirdMoment, v_thirdMoment])

    return color_feature

def IOU(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    flag = 1
    interArea = (xB - xA) * (yB - yA)
    if xB - xA < 0 and yB - yA < 0:
        interArea = -interArea
        flag = 0
    elif xB - xA < 0 or yB - yA < 0:
        flag = 0

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea*flag)

    # return the intersection over union value
    return iou

def test(boxA,arrayB):
    maxval = 0
    maxind = -1
    for index,i in enumerate(arrayB):
        thisval = IOU(boxA,(i['x'],i['y'],i['x']+i['width'],i['y']+i['height']))
        print(thisval)
        if thisval > maxval:
            maxval = thisval
            maxind = index
    return maxind,maxval

def main():
    cap = cv2.VideoCapture('../Detect_data/2min.mp4')
    # cv2.
    count = 10
    cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
    num = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print(num)
    font = cv2.FONT_HERSHEY_SIMPLEX

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        filename = '../Detect_data/Position/' + str(count) + '.json'
        with open(filename,'r') as F:
            data = json.load(F)['Players']
            # print(data)
        count += 1

        # for index,i in enumerate(data):
        #     thisframe = cv2.rectangle(frame,(i['x'],i['y']),(i['x']+i['width'],i['y']+i['height']),(255,0,0))
        #     thisframe = cv2.putText(thisframe,str(index),(i['x'],i['y']),font, 1.2, (255, 255, 255), 2)
        
        if count == 11:
            i = data[21]
            thisframe = cv2.rectangle(frame,(i['x'],i['y']),(i['x']+i['width'],i['y']+i['height']),(255,0,0))
            boxA = (i['x'],i['y'],i['x']+i['width'],i['y']+i['height'])
            thisframe = cv2.putText(thisframe,str(21),(i['x'],i['y']),font, 1.2, (255, 255, 255), 2)
        else:
            index,val = test(boxA,data)
            print(val)
            if index == -1:
                break
            i = data[index]
            thisframe = cv2.rectangle(frame,(i['x'],i['y']),(i['x']+i['width'],i['y']+i['height']),(255,0,0))
            boxA = (i['x'],i['y'],i['x']+i['width'],i['y']+i['height'])
            thisframe = cv2.putText(thisframe,str(index),(i['x'],i['y']),font, 1.2, (255, 255, 255), 2)
        print('\n')

        
        # Our operations on the frame come here
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Display the resulting frame
        cv2.imshow('frame',thisframe)
        cv2.waitKey(1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    filename1 = '../Detect_data/Images/0.jpg'
    # filename2 = '../Detect_data/Images/1000.jpg'
    img1 = cv2.imread(filename1)
    # img2 = cv2.imread(filename2)
    # fun3(img1,img2)
    filename = '../Detect_data/Position/' + str(10) + '.json'
    with open(filename,'r') as F:
        data = json.load(F)['Players']
    i = data[28]
    print(i)
    cv2.namedWindow('img',cv2.WINDOW_AUTOSIZE)
    img_crop = img1[i['y']:(i['y']+i['height']),i['x']:(i['x']+i['width'])]
    # print(img_crop)
    # cv2.rectangle(img1,(i['x'],i['y']),(i['x']+i['width'],i['y']+i['height']),(255,0,0))
    cv2.imshow('img',img_crop)
    cv2.waitKey(0)

