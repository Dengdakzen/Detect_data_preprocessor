import numpy as np
import matplotlib
# matplotlib.use('TkAgg')
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

def compare_imgs_hsv(img1,img2):  
    hsv1 = cv2.cvtColor(img1,cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(img2,cv2.COLOR_BGR2HSV)
    h1 = cv2.calcHist([hsv1], [0], None, [256], [0, 256])
    # h1 = cv2.normalize(h1,h1)
    h2 = cv2.calcHist([hsv2], [0], None, [256], [0, 256])
    s1 = cv2.calcHist([hsv1], [1], None, [256], [0, 180])
    s2 = cv2.calcHist([hsv2], [1], None, [256], [0, 180])

    r1 = cv2.compareHist(h1,h2,cv2.HISTCMP_CORREL)
    r2 = cv2.compareHist(s1,s2,cv2.HISTCMP_CORREL)
    return 0.5*r1+0.5*r2

def compare_imgs_rgb(img1,img2):  
    r1 = cv2.calcHist([img1], [0], None, [256], [0, 256])
    # r1 = cv2.normalize(h1,h1)
    r2 = cv2.calcHist([img2], [0], None, [256], [0, 256])
    g1 = cv2.calcHist([img1], [1], None, [256], [0, 180])
    g2 = cv2.calcHist([img2], [1], None, [256], [0, 180])
    b1 = cv2.calcHist([img1], [2], None, [256], [0, 180])
    b2 = cv2.calcHist([img2], [2], None, [256], [0, 180])
    # cv2.normalize()
    d1 = cv2.compareHist(r1,r2,cv2.HISTCMP_CORREL)
    d2 = cv2.compareHist(g1,g2,cv2.HISTCMP_CORREL)
    d3 = cv2.compareHist(b1,b2,cv2.HISTCMP_CORREL)
    return 0.4*d1+0.2*d2 + 0.4*d3

def generate_images_crops(img,box_array):
    images = []
    for i in box_array:
        thisimg = img[i['y']:(i['y']+i['height']),i['x']:(i['x']+i['width'])]
        images.append(thisimg)
    return images

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

def test(boxA,original_img,img,arrayB):
    images = generate_images_crops(img,arrayB)
    imgA = img[boxA[1]:boxA[3],boxA[0]:boxA[2]]
    maxval = 0
    maxind = -1
    for index,i in enumerate(arrayB):
        IOU_val = IOU(boxA,(i['x'],i['y'],i['x']+i['width'],i['y']+i['height']))
        print(IOU_val)
        hsv_hist_val = compare_imgs_hsv(imgA,images[index])
        print(hsv_hist_val)
        hsv_with_original = compare_imgs_rgb(original_img,images[index])
        print(hsv_with_original)
        val = IOU_val*0.2 + hsv_hist_val*0.3 + hsv_with_original*0.5
        print('total_val: ',val,'\n')
        if  val > maxval:
            maxval = val
            maxind = index
            final = (IOU_val,hsv_hist_val,hsv_with_original)
            final_imgs = (imgA,original_img,images[index])
    cv2.imshow('last_box',final_imgs[0])
    cv2.imshow('original_box',final_imgs[1])
    cv2.imshow('match_box',final_imgs[2])
    return maxind,maxval,final

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
        print(count)
        count += 1
        
        if count == 11:
            i = data[21]
            thisframe = cv2.rectangle(frame,(i['x'],i['y']),(i['x']+i['width'],i['y']+i['height']),(255,0,0))
            original_img = thisframe[i['y']:(i['y']+i['height']),i['x']:(i['x']+i['width'])]
            boxA = (i['x'],i['y'],i['x']+i['width'],i['y']+i['height'])
            thisframe = cv2.putText(thisframe,str(21),(i['x'],i['y']),font, 1.2, (255, 255, 255), 2)
        else:
            index,val,final = test(boxA,original_img,frame,data)
            print(final)
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
        cv2.waitKey(0)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # filename1 = '../Detect_data/Images/0.jpg'
    # # # filename2 = '../Detect_data/Images/1000.jpg'
    # img1 = cv2.imread(filename1)
    # # # img2 = cv2.imread(filename2)
    # # # fun3(img1,img2)
    # filename = '../Detect_data/Position/' + str(10) + '.json'
    # with open(filename,'r') as F:
    #     data = json.load(F)['Players']
    # i = data[28]
    # # # print(i)
    # # cv2.namedWindow('img',cv2.WINDOW_NORMAL)
    # img_crop1 = img1[i['y']:(i['y']+i['height']),i['x']:(i['x']+i['width'])]
    # # i = data[13]
    # # # print(i)
    # # cv2.namedWindow('img2',cv2.WINDOW_AUTOSIZE)
    # # img_crop2 = img1[i['y']:(i['y']+i['height']),i['x']:(i['x']+i['width'])]
    # # # print(img_crop)
    # # # cv2.rectangle(img1,(i['x'],i['y']),(i['x']+i['width'],i['y']+i['height']),(255,0,0))
    # # # cv2.imshow('img1',img_crop1)
    # # # cv2.imshow('img2',img_crop2)
    # # # cv2.waitKey(0)
    # # # fun2(img_crop)
    # # # fun1()
    # # fun3(img_crop1,img_crop2)
    # # images = generate_images_crops(img1,data)
    # # for i in images:
    # #     cv2.imshow('img',i)
    # #     cv2.waitKey(0)
    # # main()
    # hsv1 = cv2.cvtColor(img_crop1,cv2.COLOR_BGR2HSV)
    # h1 = cv2.calcHist([hsv1], [0], None, [256], [0, 256])
    # plt.plot(h1, color='r')
    # cv2.normalize(h1, h1,0,255*0.1,cv2.NORM_MINMAX)
    # # h2 = cv2.normalize(h1)
    # plt.plot(h1, color='g')
    # # h1 = cv2.normalize(h1,h1) 
    # # h2 = cv2.calcHist([hsv2], [0], None, [256], [0, 256])
    # # s1 = cv2.calcHist([hsv1], [1], None, [256], [0, 180])
    # # s2 = cv2.calcHist([hsv2], [1], None, [256], [0, 180])
    # plt.show()
    main()
