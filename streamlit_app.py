import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
from category_encoders import TargetEncoder

# Load the data
@st.cache
def load_data():
    data = pd.read_csv('BPD_arrests.csv')
    return data

data = load_data()

# Drop rows with missing target values or essential features
data = data.dropna(subset=['IncidentOffence', 'ArrestDateTime', 'Age', 'Gender', 'District'])

# Fill or drop other missing values
data['Neighborhood'] = data['Neighborhood'].fillna('Unknown')
data['ChargeDescription'] = data['ChargeDescription'].fillna('Unknown')

# Convert ArrestDateTime to datetime
data['ArrestDateTime'] = pd.to_datetime(data['ArrestDateTime'], errors='coerce')

# Create new time-based features
data['Hour'] = data['ArrestDateTime'].dt.hour
data['DayOfWeek'] = data['ArrestDateTime'].dt.dayofweek
data['Month'] = data['ArrestDateTime'].dt.month
data['TimeOfDay'] = pd.cut(data['Hour'], bins=[0, 6, 12, 18, 24], labels=['Night', 'Morning', 'Afternoon', 'Evening'], right=False)

# Encode categorical variables
cat_features = ['Gender', 'District', 'Neighborhood', 'TimeOfDay']
encoder = TargetEncoder()
# Encode the target variable first
label_encoder = LabelEncoder()
data['IncidentOffence_encoded'] = label_encoder.fit_transform(data['IncidentOffence'])

# Use the encoded target variable for TargetEncoder
data[cat_features] = encoder.fit_transform(data[cat_features], data['IncidentOffence_encoded'])

# Split the data
# Define features and target
features = ['Age', 'Gender', 'District', 'Neighborhood', 'Hour', 'DayOfWeek', 'Month', 'TimeOfDay']
target = 'IncidentOffence'

# Filter only necessary columns and drop any remaining NaNs
data_model = data[features + [target]].dropna()

# Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(data_model[features], data_model[target], test_size=0.2, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Print evaluation result
st.write(f"Model Accuracy: {accuracy:.2f}")

# Prediction form
st.header("Predict Crime Type")
with st.form("crime_prediction_form"):
    age_input = st.number_input("Offender Age", min_value=10, max_value=100, value=30)
    gender_input = st.selectbox("Gender", data['Gender'].unique())
    district_input = st.selectbox("District", data['District'].unique())
    neighborhood_input = st.selectbox("Neighborhood", data['Neighborhood'].unique())
    hour_input = st.slider("Hour of Arrest (0–23)", 0, 23, 12)
    day_of_week_input = st.selectbox("Day of Week (0=Mon)", list(range(7)))
    month_input = st.selectbox("Month", list(range(1, 13)))
    time_of_day_input = st.selectbox("Time of Day", data['TimeOfDay'].unique())

    submitted = st.form_submit_button("Predict")

    if submitted:
        input_data = pd.DataFrame({
            'Age': [age_input],
            'Gender': [gender_input],
            'District': [district_input],
            'Neighborhood': [neighborhood_input],
            'Hour': [hour_input],
            'DayOfWeek': [day_of_week_input],
            'Month': [month_input],
            'TimeOfDay': [time_of_day_input]
        })

        input_data[cat_features] = encoder.transform(input_data[cat_features])
        prediction_encoded = model.predict(input_data)[0]
        prediction_label = label_encoder.inverse_transform([prediction_encoded])[0]

        st.success(f"Predicted Crime Type: {prediction_label}")
