from flask import Flask, request, render_template
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load the trained model and one-hot encoder
model = tf.keras.models.load_model('model.keras')
onehot_encoder = joblib.load('onehot_encoder.pkl')

# Create Flask app
app = Flask(__name__)

def predict_image_with_label_and_probability(model, image_path, onehot_encoder):
    img = load_img(image_path, target_size=(128, 128)) 
    img_array = img_to_array(img) / 255.0  
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction using the model
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)

    # Convert the numerical label back to the original label using the one-hot encoder
    predicted_label = onehot_encoder.inverse_transform(predictions)
    predicted_probability = np.max(predictions)

    return predicted_label[0][0], predicted_probability  

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded.", 400

    file = request.files['file']
    if file.filename == '':
        return "No selected file.", 400

    # Save the uploaded file
    image_path = f'photos/{file.filename}'
    file.save(image_path)

    # Get prediction
    predicted_label, predicted_probability = predict_image_with_label_and_probability(model, image_path, onehot_encoder)

    return {
        'predicted_label': predicted_label,
        'predicted_probability': float(predicted_probability)
    }

if __name__ == '__main__':
    app.run(debug=True)



