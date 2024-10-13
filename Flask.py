from flask import Flask, request, render_template, send_from_directory
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf
import os

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('CNN.keras')
class_labels = ['drink', 'food', 'inside', 'menu', 'outside']

def predict_class(img_path):
    try:
        img = load_img(img_path, target_size=(128, 128))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        predicted_class = class_labels[predicted_class_index]
        predicted_probability = f"{predictions[0][predicted_class_index]:.2f}"
        
        return predicted_class, predicted_probability
    except Exception as e:
        return f"Error during prediction: {str(e)}", "0.00"

@app.route("/", methods=["GET", "POST"])
@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index.html", error="No file part")
        file = request.files["file"]
        if file.filename == "":
            return render_template("index.html", error="No selected file")

        # Create uploads directory if it doesn't exist
        uploads_dir = 'uploads'
        os.makedirs(uploads_dir, exist_ok=True)

        # Save the uploaded file temporarily
        img_path = os.path.join(uploads_dir, file.filename)
        try:
            file.save(img_path)
        except Exception as e:
            return render_template("index.html", error=f"Error saving file: {str(e)}")

        # Predict class
        predicted_class, predicted_probability = predict_class(img_path)

        # Ensure predicted_probability is a float
        predicted_probability = float(predicted_probability)

        # Return the result with the uploaded image
        return render_template("result.html", predicted_class=predicted_class, predicted_probability=predicted_probability, filename=file.filename)
    
    return render_template("index.html")


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('uploads', filename)

if __name__ == "__main__":
    app.run(debug=True)
