import cv2 as cv
from tkinter import *
from tkinter import ttk

def list_camera_options():
    camera_options = []
    cap = cv.VideoCapture(0, cv.CAP_AVFOUNDATION)

    if cap.isOpened():
        width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        cam = cap.get(cv.CAP_PROP_FOURCC)
        camera_name = f"{cam} - {width}x{height}"
        camera_options.append(camera_name)

        cap.release()
    else:
        print("No camera available")

    return camera_options

root = Tk()
root.title("FishML")

label = Label(root, text="Enhancing Fish Disease Diagnosis through Machine Learning")
label.pack()

camera_options = list_camera_options()

selected_camera = StringVar()
selected_camera.set(camera_options[0])

camera_menu = ttk.Combobox(root, textvariable=selected_camera, values=camera_options)
camera_menu.pack()

def start_camera():
    stream = cv.VideoCapture(0)

    if not stream.isOpened():
        print("Unable to open camera stream")
        return

    while True:
        ret, frame = stream.read()
        if not ret:
            print("Unable to read frame from the camera stream")
            break

        cv.imshow("Enhancing Fish Disease Diagnosis through Machine Learning", frame)
        if cv.waitKey(1) == ord('q'):
            break

    stream.release()
    cv.destroyAllWindows()

start_button = Button(root, text="START FishML", command=start_camera, bg="red")
start_button.pack()
root.mainloop()
