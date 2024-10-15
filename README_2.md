# DNN Model Deployment with Streamlit and Flask

This project involves building and deploying a Deep Neural Network (DNN) using Streamlit and Flask. The DNN model processes input features to make predictions, providing both a web interface through Streamlit and a REST API via Flask.

#### Features

- **Feature Input**: Users can input features for prediction through a web interface.
- **DNN Model**: Built using TensorFlow/Keras for general-purpose predictions.
- **Model Deployment**:
  - **Streamlit**: Serves as the user-friendly web interface.
  - **Flask**: Provides a REST API for programmatic access to predictions.
- **Real-time Prediction**: Outputs predictions based on user-provided input features.
- **Responsive UI**: A clean and intuitive interface created with Streamlit.

#### Tech Stack

- **Frontend**: Streamlit for the user interface
- **Backend**: DNN model using TensorFlow/Keras, Flask for API
- **Programming Language**: Python

#### Libraries Used

- **TensorFlow/Keras**: For building and training the DNN model
- **Streamlit**: For deploying the app and creating the user interface
- **Flask**: For creating a RESTful API for model predictions
- **NumPy**: For handling arrays and numerical operations
- **Pandas**: For data manipulation and handling feature inputs
- **Scikit-learn**: For data preprocessing and model evaluation

#### Step-by-Step Guide

### Step 1: Clone the repository

```bash
git clone https://github.com/AbhijeetStudies/Deeplearning.git
cd Deeplearning
```

### Step 2: Set up a Virtual Environment

```bash
conda create -n dnn-venv python=3.8
```

### Step 3: Activate the virtual environment

```bash
conda activate dnn-venv
```

### Step 4: Install the Requirements

```bash
pip install -r requirements.txt
```

### Step 5: Run the Streamlit app

```bash
streamlit run streamlit_app.py
```

### Step 6: Run the Flask API

```bash
python app.py
```

Once the apps are running, you can access the Streamlit UI in your browser to enter the required feature inputs and click "Predict" to get the DNN model's prediction. For API access, you can send requests to the Flask API to receive predictions programmatically.

---

#### Future Enhancements

- **Support for larger datasets and additional features**.
- **Integration with cloud-based services for model scalability**.
- **Enhanced prediction logic using more complex DNN architectures**.

---

#### Contributions

Feel free to contribute by opening issues or submitting pull requests.
