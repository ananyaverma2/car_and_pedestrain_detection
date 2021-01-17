import cv2

video = cv2.VideoCapture('car_pedestrian_detection/video2.mp4')

while True:

    happening, frame = video.read()
    grayed_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    training_car = cv2.CascadeClassifier('car_pedestrian_detection/car_detection.xml')
    training_pedestrian = cv2.CascadeClassifier('car_pedestrian_detection/haarcascade_fullbody.xml')


    coordinates_car = training_car.detectMultiScale(grayed_image, scaleFactor = 1.2)
    coordinates_pedestrian = training_pedestrian.detectMultiScale(grayed_image, scaleFactor = 1.2)

    for x,y,w,h in coordinates_car :
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    for x,y,w,h in coordinates_pedestrian :
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

    cv2.imshow('car and pedestrian detection', frame)
    key = cv2.waitKey(1)

    if key==113 or key==81:
        break

video.release()

print("Code Completed")