import cv2
import json
import numpy as np

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = (xB - xA + 1) * (yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

# filename = '../Detect_data/Position/' + str(20) + '.json'
# with open(filename,'r') as F:
#     data = json.load(F)['Players']
#     # print(data)
# k = data[10]
# j = data[10]

# print(bb_intersection_over_union((k['x'],k['y'],k['x']+k['width'],k['y']+k['height']),(j['x'] + 20,j['y']+20,j['x']+20+j['width'],j['y']+20+j['height'])))
#course15.py
import numpy as np
import cv2

# multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades

#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

eyeglasses_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')

smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

cap = cv2.VideoCapture('../Detect_data/2min.mp4')

while(cap.isOpened()):
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    smile = smile_cascade.detectMultiScale(gray)
    for (sm_x,sm_y,sm_w,sm_h) in smile:
        cv2.rectangle(gray,(sm_x,sm_y),(sm_x+sm_w,sm_y+sm_h),(0,0,255),2)
        
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img,'Eye',(ex+x,ey+y), font, 0.5, (11,255,255), 1, cv2.LINE_AA)
        #eyeglasses = eyeglasses_cascade.detectMultiScale(roi_gray)
        #for (e_gx,e_gy,e_gw,e_gh) in eyeglasses:
        #    cv2.rectangle(roi_color,(e_gx,e_gy),(e_gx+e_gw,e_gy+e_gh),(0,0,255),2)
            #roi_gray = gray[ey:ey+eh, ex:ex+ew]#
            #roi_color = img[ey:ey+eh, ex:ex+ew]#

    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    #print(k)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
print(smile)