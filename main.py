#->Aathithya & Caleb

import os
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from PIL import Image

def train_model():
    # directory for our datasets
    base_dir = './Dataset.csv' #change accordinly 
    train_dir = os.path.join(base_dir, 'training_set') #training
    validation_dir = os.path.join(base_dir, 'validation_set') #validating

    
    img_width, img_height = 150, 150 # img height and width respectively
    batch_size = 20

    # creates image data generators for datasets - applies augmentations to images (rescales/resizes images)
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    test_datagen = ImageDataGenerator(rescale=1./255) # makes ALL images 1 standard size

    # tensorflow's way to train model [both training and validation sets]
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')

    validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')

    # creates a sequence of layers in a neural network architecture (for processing 2D data)
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(img_width, img_height, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    # classification stuff
    learningrate = 0.001
    model.compile(optimizer=RMSprop(lr=learningrate), loss='binary_crossentropy', metrics=['accuracy'])

    # training stuff
    model.fit(
        train_generator,
        steps_per_epoch=100,  # 100 images used
        epochs=15,  # data is passed 15 times
        validation_data=validation_generator,
        validation_steps=50,
        verbose=2)

    # saves the TRAINED model as model_trained.h5
    model.save('model_trained.h5')
    return model

# load model
def load_trained_model(model_path='model_trained.h5'):
    if not os.path.exists(model_path):
        return train_model()
    else:
        return tf.keras.models.load_model(model_path)

def calculate_optimal_threshold(model,healthy_img_path,infected_img_path):
    # preprocess both images
    '''
    NOTE: these are NOT the images we use to train our model. they are only used to calculate a certain numerical threshold for 
    the classifican of images
    '''
    healthy_img = preprocess_image("./healthy.png") # NOTE: change path
    infected_img = preprocess_image("./infected.png") # NOTE: change path
    
    # calculate the threshold
    healthy_pred = model.predict(healthy_img) #healthy img threshold
    infected_pred = model.predict(infected_img) #infected img threshold
    
    # calculate the average of the prediction scores
    threshold = (healthy_pred[0] + infected_pred[0]) / 2
    return threshold

#->Darryan
def preprocess_image(image_file): 
    img = Image.open(image_file)
    img = img.convert('RGB')  # image to RBG format
    img = img.resize((150, 150))  # resize image
    img_array = img_to_array(img)  # converts image to array
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # normalisation
    return img_array


def main():
    st.title("Fish ML")

    model = load_trained_model()

    # path to img
    healthy_img_path = "./healthy.png" # NOTE: change path
    infected_img_path = "./infected.png" # NOTE: change path
    
    # calculates the optimal threshold
    optimal_threshold = calculate_optimal_threshold(model, healthy_img_path, infected_img_path)
    st.write(f"Optimal Threshold: {optimal_threshold}")

    image_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

    if image_file is not None:
        st.image(image_file, use_column_width=True)
        processed_image = preprocess_image(image_file)
        predictions = model.predict(processed_image)

        if predictions[0] <= optimal_threshold:
            st.write("Prediction: This fish is diseased" + ", " + str(predictions)) 
        else:
            st.write("Prediction: This fish is not diseased" + ", " + str(predictions))


if __name__ == "__main__":
    main()
