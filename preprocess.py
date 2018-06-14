import numpy as np
import cv2
import json

def bb_intersection_over_union(boxA, boxB):
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
        thisval = bb_intersection_over_union(boxA,(i['x'],i['y'],i['x']+i['width'],i['y']+i['height']))
        print(thisval)
        if thisval > maxval:
            maxval = thisval
            maxind = index
    return maxind,maxval


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

