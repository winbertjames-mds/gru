import streamlit as st
import pandas as pd
from datetime import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
import numpy as np

# Load the GRU model
model = load_model('gru_model.hdf5')

# Load the submission data
submission_data = pd.read_csv('submission2022.csv')

# Convert the 'Date' column to datetime
submission_data['Date'] = pd.to_datetime(submission_data['Date'], format='%d-%b-%y')

# Set the app title
st.title('Student Submission Predictor using GRU')

# Add a description
st.markdown('''
This app predicts the number of student submissions based on a given date of a non-profit organization.
The current limitation is that you have to input date within 11-Oct-2021 to 01-Sep-22. The accuracy is still underdevelopment.
This app is just a demonstration of deploying a model in Streamlit.
''')

# Input field for future date
input_date = st.text_input('Enter a future date in DD-MMM-YYYY format', '01-Jan-2022')

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
try:
    target_index = time_series_data.index.get_loc(input_date_str)
    input_sequence, _ = data_generator[target_index]  # Extract only the input features
    
    # Convert the tuple to a NumPy array
    input_array = np.array(input_sequence)
    
    # Reshape the input sequence for compatibility with the model
    input_features = input_array.reshape((input_array.shape[0], input_array.shape[1], 1))
    
    # Predict the submission number
    prediction = model.predict(input_features)[0][0]
    
    # Display the prediction
    st.success(f'Predicted submission number for {input_date_str}: {prediction:.2f}')

except KeyError:
    st.error(f'No data available for the input date: {input_date_str}. Please try a different date.')
