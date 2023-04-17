import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

cap = cv2.VideoCapture("../Videos/v5.mp4")  # For Video

model = YOLO("../Yolo-Weights/yolov8n.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

#mask = cv2.imread("mask.png")

# Tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
flag= False
limits = [300, 497, 973, 497]
point1=[300,497]
point2=[973,497]
totalCount = []
dcnt=1
while True:
    success, img = cap.read()
    #imgRegion = cv2.bitwise_and(img,mask)
    car_cascade = cv2.CascadeClassifier('cars.xml')

    if point1 and point2:

        # Rectangle marker
        r1 = cv2.rectangle(img, point1, point2, (100, 50, 200), 5)
        frame_ROI = img[point1[1]:point2[1], point1[0]:point2[0]]
        font = cv2.FONT_HERSHEY_SIMPLEX
        cimg = img

        # converting to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # color range--Giving range for red, green and yellow
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        lower_green = np.array([40, 50, 50])
        upper_green = np.array([90, 255, 255])
        lower_yellow = np.array([15, 150, 150])
        upper_yellow = np.array([35, 255, 255])

        # to perform basic thresholding operations to detecting specific color
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        maskg = cv2.inRange(hsv, lower_green, upper_green)
        masky = cv2.inRange(hsv, lower_yellow, upper_yellow)
        maskr = cv2.add(mask1, mask2)
        size = img.shape

        # used Hough Transform to draw circle on specific color of signal
        r_circles = cv2.HoughCircles(maskr, cv2.HOUGH_GRADIENT, 1, 80,
                                     param1=50, param2=10, minRadius=0, maxRadius=30)

        g_circles = cv2.HoughCircles(maskg, cv2.HOUGH_GRADIENT, 1, 60,
                                     param1=50, param2=10, minRadius=0, maxRadius=30)

        y_circles = cv2.HoughCircles(masky, cv2.HOUGH_GRADIENT, 1, 30,
                                     param1=50, param2=5, minRadius=0, maxRadius=30)

        # traffic light detect
        r = 5
        bound = 4.0 / 10
        if r_circles is not None:
            r_circles = np.uint16(np.around(r_circles))

            for i in r_circles[0, :]:
                if i[0] > size[1] or i[1] > size[0] or i[1] > size[0] * bound:
                    continue

                h, s = 0.0, 0.0
                for m in range(-r, r):
                    for n in range(-r, r):

                        if (i[1] + m) >= size[0] or (i[0] + n) >= size[1]:
                            continue
                        h += maskr[i[1] + m, i[0] + n]
                        s += 1
                if h / s > 50:
                    cv2.circle(cimg, (i[0], i[1]), i[2] + 10, (0, 255, 0), 2)
                    cv2.circle(maskr, (i[0], i[1]), i[2] + 30, (255, 255, 255), 2)
                    cv2.putText(cimg, 'RED', (i[0], i[1]), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
                    flag=True
                    # if drawing is False:
                    #     # convert video into gray scale of each frames
                    #     ROI_grayscale = cv2.cvtColor(frame_ROI, cv2.COLOR_BGR2GRAY)
                    #     # detect cars in the video
                    #     cars_ROI = car_cascade.detectMultiScale(ROI_grayscale, 1.1, 1)
                    #     for (x, y, w, h) in cars_ROI:
                    #         cv2.rectangle(frame_ROI, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    #         # name = 'K:\EDD\EDI dec2k20/data/frame' + str(currentframe) + '.jpg'
                    #         # print('creating...' + name)
                    #         # cv2.imwrite(name, img)
                    #         # currentframe += 1

        if g_circles is not None:
            g_circles = np.uint16(np.around(g_circles))

            for i in g_circles[0, :]:
                if i[0] > size[1] or i[1] > size[0] or i[1] > size[0] * bound:
                    continue

                h, s = 0.0, 0.0
                for m in range(-r, r):
                    for n in range(-r, r):

                        if (i[1] + m) >= size[0] or (i[0] + n) >= size[1]:
                            continue
                        h += maskg[i[1] + m, i[0] + n]
                        s += 1
                if h / s > 100:
                    cv2.circle(cimg, (i[0], i[1]), i[2] + 10, (0, 255, 0), 2)
                    cv2.circle(maskg, (i[0], i[1]), i[2] + 30, (255, 255, 255), 2)
                    cv2.putText(cimg, 'GREEN', (i[0], i[1]), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
                    flag=False

        if y_circles is not None:
            y_circles = np.uint16(np.around(y_circles))

            for i in y_circles[0, :]:
                if i[0] > size[1] or i[1] > size[0] or i[1] > size[0] * bound:
                    continue

                h, s = 0.0, 0.0
                for m in range(-r, r):
                    for n in range(-r, r):

                        if (i[1] + m) >= size[0] or (i[0] + n) >= size[1]:
                            continue
                        h += masky[i[1] + m, i[0] + n]
                        s += 1
                if h / s > 50:
                    cv2.circle(cimg, (i[0], i[1]), i[2] + 10, (0, 255, 0), 2)
                    cv2.circle(masky, (i[0], i[1]), i[2] + 30, (255, 255, 255), 2)
                    cv2.putText(cimg, 'YELLOW', (i[0], i[1]), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
                    flag=True

                    # ------------------Detect car ROI-------------------#
    #imgGraphics = cv2.imread("graphics.png", cv2.IMREAD_UNCHANGED)
    #img = cvzone.overlayPNG(img, imgGraphics, (0, 0))
    results = model(img, stream=True)

    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            w, h = x2 - x1, y2 - y1

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if currentClass == "car" or currentClass == "truck" or currentClass == "bus" \
                    or currentClass == "motorbike" and conf > 0.3:
                # cvzone.putTextRect(img, f'{currentClass} {conf}', (max(0, x1), max(35, y1)),
                #                    scale=0.6, thickness=1, offset=3)
                # cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=5)
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    resultsTracker = tracker.update(detections)

    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)
    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(result)
        print(flag)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
        cvzone.putTextRect(img, f' {int(id)}', (max(0, x1), max(35, y1)),
                           scale=2, thickness=3, offset=10)

        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        if limits[0] < cx < limits[2] and limits[1] - 15 < cy < limits[1] + 15 and flag==True:
            if totalCount.count(id) == 0:
                totalCount.append(id)
                cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(0, 0, 255))
                cimg = img[y1:y2,x1:x2]
                print("cimg",cimg)
                cv2.imwrite("Detected Images/violation_" + str(dcnt) + ".jpg", cimg)
                dcnt = dcnt + 1
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)

    # cvzone.putTextRect(img, f' Count: {len(totalCount)}', (50, 50))
    cv2.putText(img,str(len(totalCount)),(255,100),cv2.FONT_HERSHEY_PLAIN,5,(50,50,255),8)

    cv2.imshow("Image", img)
    # cv2.imshow("ImageRegion", imgRegion)
    cv2.waitKey(1)
