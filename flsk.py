from flask import Flask, request, jsonify
from geopy.geocoders import Nominatim
import joblib  # Assuming you saved your model using joblib
import numpy as np 

app = Flask(__name__)

# Load your trained model
model = joblib.load('./x06.joblib')

# Initialize geolocator
geolocator = Nominatim(user_agent="parking_predictor")

@app.route('/predict', methods=['POST'])
def predict():
    # Get the JSON data from the request
    data = request.get_json()
    print(f"Received data: {data}")  # Print received data for debugging
    
    # Check for 'address' in the request data
    address = data.get('address')
    if not address:
        return jsonify({'error': 'Address not provided'}), 400

    # Trim any leading/trailing whitespace from the address
    address = address.strip()

    # Convert address to latitude and longitude
    location = geolocator.geocode(address)
    if location is None:
        print(f"Geocode failed for address: {address}")  # Debug output for failed geocode
        return jsonify({'error': 'Address not found'}), 404

    latitude = location.latitude
    longitude = location.longitude
    
    print(f"Latitude: {latitude}, Longitude: {longitude}")  # Debug output
    
    # Prepare input for the model (make sure to shape it correctly)
    input_data = np.array([[latitude, longitude]])

    # Make prediction
    prediction = model.predict(input_data)

    # Assuming binary classification, ensure the prediction is rounded and converted to int
    predicted_class = int(np.round(prediction[0]))  # Change [0][0] to [0] if your model returns a single output
    return jsonify({'prediction': predicted_class})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5089)
