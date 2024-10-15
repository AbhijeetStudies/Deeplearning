# import streamlit as st
# import requests

# st.title("Parking Availability Predictor")

# address = st.text_input("Enter an address:")

# if st.button("Check Availability"):
#     if address:
#         # Call the Flask API
#         response = requests.post('http://localhost:5000/predict', json={'address': address})
        
#         if response.status_code == 200:
#             prediction = response.json()['prediction']
#             availability = "Open Parking" if prediction == 1 else "Closed Parking"
#             st.success(f"Prediction: {availability}")
#         else:
#             st.error("Error: " + response.json().get('error', 'Unknown error'))
#     else:
#         st.warning("Please enter an address.")
import streamlit as st
import requests

st.title("Parking Availability Predictor")

address = st.text_input("Enter an address:")

if st.button("Check Availability"):
    if address:
        # Call the Flask API
        response = requests.post('http://localhost:5089/predict', json={'address': address})
        
        if response.status_code == 200:
            prediction = response.json()['prediction']
            availability = "Open Parking" if prediction == 1 else "Closed Parking"
            st.success(f"Prediction: {availability}")
        else:
            # Print the raw response text for    debugging
            st.error(f"Error: {response.status_code} - {response.text}")
    else:
        st.warning("Please enter an address.")
