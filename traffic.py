from roboflow import Roboflow
import numpy as np
from sort import *
import cvzone
import cv2
def violation():
    rf = Roboflow(api_key="YOUR_API_KEY")
    project = rf.workspace().project("YOUR_PROJECT_NAME")
    model = project.version(1).model #Model number of your project
    tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
    totalCount = []
    limits=[]
    flag=False
    cap=cv2.VideoCapture("videos/caught.mp4")
    check=0
    dcnt=1
    while True:
        ret,frame=cap.read()
        response_json=model.predict(frame, confidence=40, overlap=30).json()
        predictions = response_json['predictions']
        
        detections = np.empty((0, 5))
        # Draw bounding boxes on the image
        for prediction in predictions:
            x = int(prediction['x'])
            y = int(prediction['y'])
            width = int(prediction['width'])
            height = int(prediction['height'])
            class_name = prediction['class']

            x1 = x - width // 2
            y1 = y - height // 2
            x2 = x + width // 2
            y2 = y + height // 2

            # Draw the adjusted bounding box rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Put class label text
            label = f"{class_name}: {prediction['confidence']:.2f}"
            if class_name=="zebra_crossing" and check==0:
                limits=[x1,y1,x2,y2]
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                check=1
            if class_name=="red_light":
                print("Red light")
                flag=True
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            if class_name=="green_light":
                flag=False
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            if class_name=="car" or class_name=="bike" or class_name=="truck":
                currentArray = np.array([x1, y1, x2, y2, prediction['confidence']])
                detections = np.vstack((detections, currentArray))
        
        resultsTracker = tracker.update(detections)
        cv2.line(frame, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)
        for result in resultsTracker:
            x1, y1, x2, y2, id = result
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            print(result)
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(frame, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
            cvzone.putTextRect(frame, f' {int(id)}', (max(0, x1), max(35, y1)),
                               scale=2, thickness=3, offset=10)

            cx, cy = x1 + w // 2, y1 + h // 2
            cv2.circle(frame, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
            if limits[0] < cx < limits[2] and limits[1] - 5 < cy < limits[1] + 5 and flag==True:
                if totalCount.count(id) == 0:
                    totalCount.append(id)
                    cvzone.cornerRect(frame, (x1, y1, w, h), l=9, rt=2, colorR=(0, 0, 255))
                    cimg = frame[y1:y2,x1:x2]
                    # print("cimg",cimg)
                    cv2.imwrite("Detected Images/violation_" + str(dcnt) + ".jpg", cimg)
                    dcnt = dcnt + 1
                    cv2.line(frame, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)



        cv2.putText(frame,str(len(totalCount)),(255,100),cv2.FONT_HERSHEY_PLAIN,5,(50,50,255),8)
        cv2.imshow("My project",frame)


        if(cv2.waitKey(30)==27):
            break
    cap.release()
    cv2.destroyAllWindows()
# if __name__=="__main__":
#     main()