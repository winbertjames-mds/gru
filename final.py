import streamlit as st
import pandas as pd
from datetime import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

# Load the GRU model
model = load_model('gru_model.hdf5')

# Load the submission data
submission_data = pd.read_csv('submission2022.csv')

# Convert the 'Date' column to datetime
submission_data['Date'] = pd.to_datetime(submission_data['Date'], format='%d-%b-%y')

# Set the app title
st.title('Student Submission Predictor')

# Input field for date
input_date = st.text_input('Enter a specific date in DD-MMM-YYYY format', '01-Jan-2023')

# Parse the input date
try:
    parsed_date = datetime.strptime(input_date, '%d-%b-%Y')
    input_date_str = parsed_date.strftime('%Y-%m-%d')
except ValueError:
    st.error('Invalid date format. Please enter a date in %d-%b-%Y format.')
    st.stop()

# Filter submissions for the given date
filtered_data = submission_data[submission_data['Date'] == input_date_str]

if len(filtered_data) > 0:
    # Get the features for the input date
    input_features = filtered_data.drop(['Date', 'Submission'], axis=1).values
    
    # Reshape the input features for compatibility with the model
    input_features = input_features.reshape((input_features.shape[0], input_features.shape[1], 1))

    # Create a dummy target variable for TimeseriesGenerator
    target_dummy = [0] * len(input_features)

    # Generate sequences using TimeseriesGenerator
    data_generator = TimeseriesGenerator(input_features, target_dummy, length=5, batch_size=1)

    # Predict the submission number
    prediction = model.predict(data_generator)[0][0]

    # Display the prediction
    st.success(f'Predicted submission number for {input_date_str}: {prediction:.2f}')
else:
    st.info('No submissions found for the given date.')
