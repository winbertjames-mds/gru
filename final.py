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

# Input field for future date
input_date = st.text_input('Enter a future date in DD-MMM-YYYY format', '01-Jan-2024')

# Parse the input date
try:
    parsed_date = datetime.strptime(input_date, '%d-%b-%Y')
    input_date_str = parsed_date.strftime('%Y-%m-%d')
except ValueError:
    st.error('Invalid date format. Please enter a date in %d-%b-%Y format.')
    st.stop()

# Convert the submission data to a time series dataset
time_series_data = pd.Series(submission_data['Submission'].values, index=submission_data['Date'])

# Generate sequences using TimeseriesGenerator for the entire dataset
data_generator = TimeseriesGenerator(time_series_data.values, time_series_data.values, length=5, batch_size=1)

# Find the sequence that corresponds to the input date
target_index = time_series_data.index.get_loc(input_date_str)
input_sequence = data_generator[target_index]

# Reshape the input sequence for compatibility with the model
input_features = input_sequence.reshape((input_sequence.shape[0], input_sequence.shape[1], 1))

# Predict the submission number
prediction = model.predict(input_features)[0][0]

# Display the prediction
st.success(f'Predicted submission number for {input_date_str}: {prediction:.2f}')
