import cv2 as cv

stream = cv.VideoCapture(0)

if not stream.isOpened():
    print("Unable to open screen")
    exit
while(True):
    ret, frame = stream.read()
    if not ret:
        print("Unable to open stream")
        break

    cv.imshow("Enhancing Fish Disease Diagnosis through Machine Learning", frame)
    if cv.waitKey(1) == ord('q'):
        break
stream.release()
cv.destroyAllWindows()