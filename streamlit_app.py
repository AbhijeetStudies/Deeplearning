import streamlit as st
import joblib
import tensorflow as tf
from PIL import Image
import numpy as np

@st.cache_data
def load_model():
    # Load your trained model here
    model = tf.keras.models.load_model('model.keras')
    return model

@st.cache_data
def load_label_encoder():
    # Load the one-hot encoder from the file
    onehot_encoder = joblib.load('onehot_encoder.pkl')
    return onehot_encoder

model = load_model()
onehot_encoder = load_label_encoder()

st.title("Image Classification using DNN")

# Uploading the image
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image for model prediction
    img = image.resize((128, 128))  # Resize as per your model input size
    img_array = np.array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict the label
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction, axis=1)  # Get the index of the predicted class

    # Inverse transform to get the actual class label
    predicted_class = onehot_encoder.inverse_transform(prediction)

    # Get the predicted probability for the predicted class
    predicted_probability = np.max(prediction)  # The highest probability

    # Display the results
    st.write(f"Predicted Class: {predicted_class[0][0]}")
    st.write(f"Probability of Predicted Class: {predicted_probability:.4f}")