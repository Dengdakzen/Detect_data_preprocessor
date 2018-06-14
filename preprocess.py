import numpy as np
import cv2
import json

cap = cv2.VideoCapture('../Detect_data/2min.mp4')
# cv2.
count = 0

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    filename = '../Detect_data/Position/' + str(count) + '.json'
    with open(filename,'r') as F:
        data = json.load(F)['Players']
        # print(data)
    count += 1

    for i in data:
        thisframe = cv2.rectangle(frame,(i['x'],i['y']),(i['x']+i['width'],i['y']+i['height']),(255,0,0))
    
    # Our operations on the frame come here
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame',thisframe)
    cv2.waitKey(30)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()