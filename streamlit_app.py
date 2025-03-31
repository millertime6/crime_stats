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

#create a normal distribution of the data using age 
sns.histplot(data_model['Age'], kde=True)
st.pyplot()

# build some summary statistics
st.write(data_model.describe())
print(data_model.describe())

X = data_model[features]
y = data_model[target]

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


