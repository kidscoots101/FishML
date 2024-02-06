import cv2 as cv
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

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
        st.write("No camera available")

    return camera_options

st.title("FishML")
st.write("Enhancing Fish Disease Diagnosis through Machine Learning")

# START of ML part

model = Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

def load_and_prepare_image(image_path):
    img = Image.open(image_path).convert('RGB')  
    img = img.resize((150, 150))  
    img_array = image.img_to_array(img)  
    return img_array

infected_img = load_and_prepare_image('/Users/caleb/developer/fish-disease-diagnosis_py/unhealthy.png') # NOTE: make sure to update file
healthy_img = load_and_prepare_image('/Users/caleb/developer/fish-disease-diagnosis_py/healthy.png')  # NOTE: make sure to update file

train_images = np.stack([infected_img, healthy_img])

train_labels = np.array([1, 0])

model.fit(train_images, train_labels, epochs=10)


# END of ML part


agree_to_terms = st.checkbox("I agree that the developer is not responsible for any death as a result of this product.")
if agree_to_terms:
    choice = st.selectbox("Pick a method", ["Video", "Upload Picture", "Take picture"])
    c1, c2= st.columns(2)
    if choice == "Video":
        camera_options = list_camera_options()
        selected_camera = st.selectbox("Select Camera", camera_options)

        start_button_key = "start_button_key"
        start_button = st.button("START FishML", key=start_button_key)
        stream = cv.VideoCapture(0)
        FRAME_WINDOW = st.image([])
        camera = cv.VideoCapture(0)
        stop_button_key = "stop_button_key"
        if start_button:
            if not stream.isOpened():
                st.error("Unable to open camera stream")
            else:
                stop_button = st.button("STOP FishML", key=stop_button_key)        
                while True:
                    _, frame = camera.read()
                    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                    FRAME_WINDOW.image(frame)
                    if stop_button:
                        break
    elif choice == "Upload Picture":
        # ensure only 1 image uploaded at a time
        uploaded_file = st.file_uploader("Upload picture")
        c1, c2= st.columns(2)
        if uploaded_file is not None:
            c1.image(uploaded_file)
            uploaded_image = Image.open(uploaded_file).convert('RGB').resize((150, 150))  # converted image to RGB and resize
            image_array = image.img_to_array(uploaded_image)
            image_array = np.expand_dims(image_array, axis=0)
            
            prediction = model.predict(image_array)
    
            prediction_text = "The fish is likely healthy." if prediction < 0.5 else "The fish is likely diseased."
    
            with c1:
                st.write(prediction_text)
    elif choice == "Take picture":
        uploaded_file = st.camera_input("Take a picture")
        c1, c2= st.columns(2)
        if uploaded_file is not None: 
           c1.image(uploaded_file)
           uploaded_image = Image.open(uploaded_file).convert('RGB').resize((150, 150))  # converted image to RGB and resize
           image_array = image.img_to_array(uploaded_image)
           image_array = np.expand_dims(image_array, axis=0)
            
           prediction = model.predict(image_array)
           if prediction < 0.5:
            st.write("The fish is likely healthy.")
           else:
            st.write("The fish is likely diseased.")
else:
    st.text("Please agree to our terms before proceeding.")