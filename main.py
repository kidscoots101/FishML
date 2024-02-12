# Necessary imports for the project
import os
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from PIL import Image

# Defines the function to train a model for disease detection from images
def train_model():
    # The base directory for training and validation sets
    base_dir = '/Users/aathithyaj/Desktop/GitHub/fish-disease-diagnosis-py/Dataset.csv' #change accordinly 
    train_dir = os.path.join(base_dir, 'testing_set') 
    validation_dir = os.path.join(base_dir, 'validation_set')

    # Set the image dimensions and batch size for training
    img_width, img_height = 150, 150
    batch_size = 20

    # Initialize ImageDataGenerators for training and validation sets with data augmentation for the training set
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    test_datagen = ImageDataGenerator(rescale=1./255)

    # Configure train and validation generators
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

    # Define the model architecture
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

    # Compile the model with RMSprop optimizer
    model.compile(optimizer=RMSprop(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model with the training and validation data
    model.fit(
        train_generator,
        steps_per_epoch=100,  # Defines how many batches of images to use in each epoch
        epochs=15,  # Defines how many times the entire dataset is passed through the network
        validation_data=validation_generator,
        validation_steps=50,  # Defines how many validation batches to process before stopping each epoch
        verbose=2)

    # Save the trained model to disk
    model.save('model_trained.h5')
    return model

# Function to load the trained model, training if necessary
def load_trained_model(model_path='model_trained.h5'):
    # Check if the model file exists; train the model if it doesn't
    if not os.path.exists(model_path):
        return train_model()
    else:
        # Load and return the trained model if it exists
        return tf.keras.models.load_model(model_path)

# Function to preprocess the uploaded image file
def preprocess_image(image_file):
    # Open the uploaded image file
    img = Image.open(image_file)
    img = img.convert('RGB')  # Convert image to RGB format
    img = img.resize((150, 150))  # Resize image to match model's expected input dimensions
    img_array = img_to_array(img)  # Convert the image to an array
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize the image
    return img_array

# Main function for the Streamlit app
def main():
    st.title("Disease Detection from Images")

    # Load the trained model
    model = load_trained_model()

    # Create a file uploader widget in Streamlit
    image_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

# Display the uploaded image and predict its class
    if image_file is not None:
        st.image(image_file, use_column_width=True)  # Display the uploaded image
        processed_image = preprocess_image(image_file)  # Preprocess the uploaded image
        predictions = model.predict(processed_image)  # Make a prediction using the model

        # Display the prediction result in Streamlit
        if predictions[0] >= 0.265: #0.265 is due to current displayed balues
            st.write("Prediction: Diseased" + ", " + str(predictions)) #Prediction
        else:
            st.write("Prediction: Not Diseased" + ", " + str(predictions)) #Prediction

if name == "main":
    main()