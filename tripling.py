from roboflow import Roboflow
import numpy as np
import cv2
from sort import *



def detect():
    rf = Roboflow(api_key="YOUR_API_KEY")
    project = rf.workspace().project("YOUR_PROJECT_NAME")
    model = project.version(2).model #Model number of your project
    cap=cv2.VideoCapture("videos/sample_video.mp4")
    tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
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
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            if class_name=="offender":
                currentArray = np.array([x1, y1, x2, y2, prediction['confidence']])
                detections = np.vstack((detections, currentArray))
        resultsTracker = tracker.update(detections)
        for result in resultsTracker:
            x1, y1, x2, y2, id = result
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            print(result)
            w, h = x2 - x1, y2 - y1
            cimg = frame[y1:y2,x1:x2]
            cv2.imwrite("Violation Images/violation_" + str(dcnt) + ".jpg", cimg)
            dcnt = dcnt + 1

        cv2.imshow("My project",frame)


        if(cv2.waitKey(30)==27):
            break
    cap.release()
    cv2.destroyAllWindows()

# if __name__=="__main__":
#     main()
