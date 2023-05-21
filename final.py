import streamlit as st
import pandas as pd
from datetime import datetime
from tensorflow.keras.models import load_model

# Load the GRU model
model = load_model('gru_model.hdf5')

# Load the submission data
submission_data = pd.read_csv('submission2023.csv')

# Convert the 'Date' column to datetime
submission_data['Date'] = pd.to_datetime(submission_data['Date'], format='%d-%b-%y')

# Set the app title
st.title('Student Submission Predictor')

# Input field for date
input_date = st.text_input('Enter a specific date in %d-%b-%Y format', '01-Jan-2023')

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

    # Predict the submission number
    prediction = model.predict(input_features)[0][0]

    # Display the prediction
    st.success(f'Predicted submission number for {input_date_str}: {prediction:.2f}')
else:
    st.info('No submissions found for the given date.')
