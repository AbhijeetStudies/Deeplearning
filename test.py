import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
import joblib

# Set directories
image_folder = "C:/Users/singh/Downloads/yelp_dataset/photos"
encoder_path = "path/to/save/encoder.pkl"

# Load image metadata
photos_df = pd.read_json("C:/Users/singh/Downloads/yelp_dataset/photos.json", lines=True)

# Load images with metadata
def load_images_with_metadata(photo_df, image_folder):
    images, labels = [], []
    
    for index, row in photo_df.iterrows():
        photo_id = row['photo_id']
        label = row['label']
        
        img_path = os.path.join(image_folder, f"{photo_id}.jpg")
        try:
            img = load_img(img_path, target_size=(128, 128))
            img_array = img_to_array(img) / 255.0
            images.append(img_array)
            labels.append(label)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")

    return np.array(images), np.array(labels)

# Load the test dataset
X_test, y_test = load_images_with_metadata(photos_df, image_folder)

# Load the encoder
if os.path.exists(encoder_path):
    encoder = joblib.load(encoder_path)
    y_test_onehot = encoder.transform(y_test.reshape(-1, 1))
else:
    raise FileNotFoundError(f"Encoder not found at {encoder_path}. Please run the training script to save the encoder.")

# Load the trained model
model = tf.keras.models.load_model('best_model.keras')

# Evaluate the model
y_test_pred = model.predict(X_test)
y_test_pred_classes = np.argmax(y_test_pred, axis=1)

# Classification report
print("Classification Report:")
from sklearn.metrics import classification_report
print(classification_report(np.argmax(y_test_onehot, axis=1), y_test_pred_classes))
