#->Aathithya & Caleb

import os
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from PIL import Image

def train_model():
    # directory for our datasets
    base_dir = './Dataset' # change accordingly 
    train_dir = os.path.join(base_dir, 'training_set') # training
    validation_dir = os.path.join(base_dir, 'testing_set') # validating

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
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

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
    healthy_img_path = "./healthy.jpeg" # NOTE: change path
    infected_img_path = "./infected.jpg" # NOTE: change path
    
    st.write(f"Classification of infected or healthy fishes")
    st.write(f"Diseases include White tailed disease, several parasitic diseases, Saprolegniasis, Bacterial gill disease, Bacterial red disease and Aeromoniasis")

    image_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

    if image_file is not None:
        st.image(image_file, use_column_width=True)
        processed_image = preprocess_image(image_file)
        predictions = model.predict(processed_image)

        if predictions[0] > 0.5:
            st.write("Prediction: This fish is infected" + ", " + str(predictions)) 
        else:
            st.write("Prediction: This fish is healthy" + ", " + str(predictions))


if __name__ == "__main__":
    main()
