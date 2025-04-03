import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import sys
from datetime import datetime
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectFromModel, RFECV
from category_encoders import TargetEncoder
import xgboost as xgb
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('crime_prediction_app')

# Page config for a cleaner look
st.set_page_config(page_title="Crime Prediction App", layout="wide")

# Function to diagnose data issues
def diagnose_data_issues(df, target_col):
    """
    Diagnose common issues in the dataset that might affect model performance
    """
    issues = []
    
    # Check for target class imbalance
    class_counts = df[target_col].value_counts()
    rare_classes = class_counts[class_counts < 5].index.tolist()
    if len(rare_classes) > 0:
        issues.append({
            'issue': 'Class Imbalance',
            'description': f'Found {len(rare_classes)} classes with fewer than 5 samples',
            'recommendation': 'Consider removing rare classes or using techniques like SMOTE or class weights'
        })
    
    # Check for missing values
    missing_cols = df.columns[df.isnull().any()].tolist()
    if missing_cols:
        issues.append({
            'issue': 'Missing Values',
            'description': f'Found missing values in columns: {", ".join(missing_cols)}',
            'recommendation': 'Consider imputation strategies or dropping columns with too many missing values'
        })
    
    # Check for high cardinality categorical features
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    high_card_cols = [col for col in cat_cols if df[col].nunique() > 50]
    if high_card_cols:
        issues.append({
            'issue': 'High Cardinality Features',
            'description': f'Columns with many unique values: {", ".join(high_card_cols)}',
            'recommendation': 'Consider target encoding or embedding techniques for these features'
        })
    
    # Check for data leakage potential
    timestamp_cols = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]
    if timestamp_cols:
        issues.append({
            'issue': 'Potential Time-based Data Leakage',
            'description': f'Found time-related columns: {", ".join(timestamp_cols)}',
            'recommendation': 'Ensure proper train-test splitting by time and avoid using future information'
        })
    
    return issues

# Load the data
@st.cache_data
def load_data():
    data = pd.read_csv('BPD_arrests.csv')
    return data

# Add a sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Data Exploration", "Model Training", "Prediction"])

# Initialize session state for storing model and encoders
if 'model' not in st.session_state:
    st.session_state.model = None
if 'encoders' not in st.session_state:
    st.session_state.encoders = {}
if 'label_encoder' not in st.session_state:
    st.session_state.label_encoder = None
if 'feature_names' not in st.session_state:
    st.session_state.feature_names = None
if 'model_metrics' not in st.session_state:
    st.session_state.model_metrics = {}

# Load data
data = load_data()

# Main function for data preprocessing
def preprocess_data(data, for_training=True):
    # Make a copy to avoid modifying the original
    df = data.copy()
    
    # Drop rows with missing target values or essential features
    df = df.dropna(subset=['IncidentOffence', 'ArrestDateTime', 'Age', 'Gender', 'District'])
    
    # Strip whitespace from relevant columns
    string_columns = df.select_dtypes(include=['object']).columns
    for col in string_columns:
        if col in df.columns:
            df[col] = df[col].str.strip() if hasattr(df[col], 'str') else df[col]
    
    # Fill missing values
    df['Neighborhood'] = df['Neighborhood'].fillna('Unknown')
    df['ChargeDescription'] = df['ChargeDescription'].fillna('Unknown')
    
    # Convert ArrestDateTime to datetime
    df['ArrestDateTime'] = pd.to_datetime(df['ArrestDateTime'], errors='coerce')
    
    # Create new time-based features
    df['Hour'] = df['ArrestDateTime'].dt.hour
    df['DayOfWeek'] = df['ArrestDateTime'].dt.dayofweek
    df['Month'] = df['ArrestDateTime'].dt.month
    df['Year'] = df['ArrestDateTime'].dt.year
    df['TimeOfDay'] = pd.cut(df['Hour'], bins=[0, 6, 12, 18, 24], labels=['Night', 'Morning', 'Afternoon', 'Evening'], right=False)
    df['Weekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)  # 5=Sat, 6=Sun
    
    # Create interaction features
    df['Age_TimeOfDay'] = df['Age'].astype(str) + "_" + df['TimeOfDay'].astype(str)
    df['District_TimeOfDay'] = df['District'].astype(str) + "_" + df['TimeOfDay'].astype(str)
    df['Gender_Age'] = df['Gender'].astype(str) + "_" + pd.cut(df['Age'], bins=[0, 18, 30, 50, 100], labels=['Minor', 'Young', 'Middle', 'Senior']).astype(str)
    
    # Store original values
    if for_training:
        st.session_state.district_options = sorted(df['District'].unique())
        st.session_state.neighborhood_options = sorted(df['Neighborhood'].unique())
        
    return df

if page == "Data Exploration":
    st.title("Data Exploration")
    
    # Display basic information
    st.subheader("Dataset Overview")
    st.write(f"Total records: {len(data)}")
    st.write(f"Columns: {', '.join(data.columns)}")
    
    # Data diagnosis
    if st.checkbox("Run Data Diagnosis"):
        issues = diagnose_data_issues(data, 'IncidentOffence')
        if issues:
            st.subheader("Potential Data Issues")
            for i, issue in enumerate(issues):
                with st.expander(f"{i+1}. {issue['issue']}"):
                    st.write(f"**Description:** {issue['description']}")
                    st.write(f"**Recommendation:** {issue['recommendation']}")
        else:
            st.success("No major issues detected in the dataset.")
    
    # Show sample data
    st.subheader("Sample Data")
    st.dataframe(data.head())
    
    # Basic statistics
    st.subheader("Summary Statistics")
    st.dataframe(data.describe())
    
    # Data preprocessing for exploration
    processed_data = preprocess_data(data)
    
    # Distribution of crime types
    st.subheader("Crime Type Distribution")
    crime_counts = processed_data['IncidentOffence'].value_counts().reset_index()
    crime_counts.columns = ['Crime Type', 'Count']
    
    # Only show top 15 for clarity
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=crime_counts.head(15), x='Count', y='Crime Type', ax=ax)
    plt.tight_layout()
    st.pyplot(fig)
    
    # Time of day analysis
    st.subheader("Crimes by Time of Day")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(data=processed_data, x='TimeOfDay', ax=ax)
    plt.tight_layout()
    st.pyplot(fig)
    
    # District analysis
    st.subheader("Crimes by District")
    district_crime = processed_data.groupby('District')['IncidentOffence'].count().sort_values(ascending=False).reset_index()
    district_crime.columns = ['District', 'Count']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=district_crime, x='Count', y='District', ax=ax)
    plt.tight_layout()
    st.pyplot(fig)
    
    # Age distribution
    st.subheader("Age Distribution of Offenders")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data=processed_data, x='Age', bins=20, kde=True, ax=ax)
    plt.tight_layout()
    st.pyplot(fig)

elif page == "Model Training":
    st.title("Model Training and Evaluation")
    
    # Process data for modeling
    processed_data = preprocess_data(data)
    
    # Add option to filter rare classes
    st.subheader("Class Filtering")
    min_samples = st.slider("Minimum samples per offense type", 1, 20, 5, 
                          help="Exclude offense types with fewer examples than this threshold")
    
    # Count offense frequencies
    offense_counts = processed_data['IncidentOffence'].value_counts()
    rare_offenses = offense_counts[offense_counts < min_samples].index.tolist()
    common_offenses = offense_counts[offense_counts >= min_samples].index.tolist()
    
    # Filter data to keep only common offenses
    processed_data_filtered = processed_data[processed_data['IncidentOffence'].isin(common_offenses)]
    
    # Display statistics
    st.info(f"Filtered out {len(rare_offenses)} rare offense types, keeping {len(common_offenses)} common types.")
    st.info(f"Dataset size reduced from {len(processed_data)} to {len(processed_data_filtered)} records ({(len(processed_data_filtered)/len(processed_data)*100):.1f}% of original data).")
    
    # Show class distribution before and after filtering
    if st.checkbox("Show offense distribution before and after filtering"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Before Filtering")
            top_offenses_before = offense_counts.head(10)
            fig1, ax1 = plt.subplots(figsize=(10, 8))
            sns.barplot(x=top_offenses_before.values, y=top_offenses_before.index, ax=ax1)
            ax1.set_title("Top 10 Offense Types (Before)")
            ax1.set_xlabel("Count")
            plt.tight_layout()
            st.pyplot(fig1)
            
        with col2:
            st.subheader("After Filtering")
            offense_counts_after = processed_data_filtered['IncidentOffence'].value_counts()
            top_offenses_after = offense_counts_after.head(10)
            fig2, ax2 = plt.subplots(figsize=(10, 8))
            sns.barplot(x=top_offenses_after.values, y=top_offenses_after.index, ax=ax2)
            ax2.set_title("Top 10 Offense Types (After)")
            ax2.set_xlabel("Count")
            plt.tight_layout()
            st.pyplot(fig2)
    
    # Use the filtered dataset for modeling
    processed_data = processed_data_filtered
    
    # Define features and target
    categorical_features = ['Gender', 'District', 'Neighborhood', 'TimeOfDay', 'Age_TimeOfDay', 'District_TimeOfDay', 'Gender_Age']
    numerical_features = ['Age', 'Hour', 'DayOfWeek', 'Month', 'Year', 'Weekend']
    
    # Store for later use
    all_features = numerical_features + categorical_features
    st.session_state.feature_names = all_features
    
    # Create feature selection options
    st.subheader("Feature Selection")
    selected_features = st.multiselect("Select features to use", all_features, default=all_features[:8])
    
    if len(selected_features) < 2:
        st.warning("Please select at least two features.")
    else:
        # Choose algorithm
        st.subheader("Model Selection")
        algorithm = st.selectbox("Select Algorithm", ["Random Forest", "XGBoost", "LightGBM"])
        
        # Choose balancing strategy
        balancing_method = st.radio("Class Balancing Method", 
                                   ["None", "SMOTE (for moderately imbalanced data)", 
                                    "RandomOverSampler (for highly imbalanced data)",
                                    "Class Weights (for highly imbalanced data)"])
        balancing = balancing_method != "None"
        
        # Hyperparameter options
        st.subheader("Hyperparameters")
        if algorithm == "Random Forest":
            n_estimators = st.slider("Number of Trees", 50, 500, 100, 50)
            max_depth = st.slider("Max Depth", 3, 30, 10, 1)
            min_samples_split = st.slider("Min Samples Split", 2, 10, 2, 1)
        elif algorithm == "XGBoost":
            n_estimators = st.slider("Number of Trees", 50, 500, 100, 50)
            max_depth = st.slider("Max Depth", 3, 15, 6, 1)
            learning_rate = st.slider("Learning Rate", 0.01, 0.3, 0.1, 0.01)
        else:  # LightGBM
            n_estimators = st.slider("Number of Trees", 50, 500, 100, 50)
            num_leaves = st.slider("Num Leaves", 20, 200, 31, 10)
            learning_rate = st.slider("Learning Rate", 0.01, 0.3, 0.1, 0.01)
        
        # Feature encoding strategy
        encoding_strategy = st.radio("Categorical Encoding Strategy", 
                                     ["One-Hot Encoding", "Target Encoding", "Combination"])
        
        # Train button
        if st.button("Train Model"):
            with st.spinner("Training model... This may take a moment."):
                # Prepare the data
                X = processed_data[selected_features]
                y = processed_data['IncidentOffence']
                
                # Create a label encoder for the target
                label_encoder = LabelEncoder()
                y_encoded = label_encoder.fit_transform(y)
                st.session_state.label_encoder = label_encoder
                
                # Split the data
                X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
                
                # Define preprocessor based on selected strategy
                if encoding_strategy == "One-Hot Encoding":
                    # For categorical features with low cardinality, use one-hot
                    low_cardinality_features = [f for f in categorical_features if f in selected_features and 
                                             X[f].nunique() < 10]
                    high_cardinality_features = [f for f in categorical_features if f in selected_features and 
                                              f not in low_cardinality_features]
                    
                    # Create preprocessing steps
                    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
                    preprocessor = ColumnTransformer(
                        transformers=[
                            ('num', SimpleImputer(strategy='median'), [f for f in numerical_features if f in selected_features]),
                            ('cat', categorical_transformer, low_cardinality_features)
                        ],
                        remainder='passthrough'
                    )
                    
                    # Store encoders for prediction
                    st.session_state.encoders['preprocessor'] = preprocessor
                    st.session_state.encoders['strategy'] = encoding_strategy
                    st.session_state.encoders['low_cardinality'] = low_cardinality_features
                    st.session_state.encoders['high_cardinality'] = high_cardinality_features
                    
                elif encoding_strategy == "Target Encoding":
                    # Use target encoding for all categorical features
                    cat_features = [f for f in categorical_features if f in selected_features]
                    num_features = [f for f in numerical_features if f in selected_features]
                    
                    # Create encoder
                    target_encoder = TargetEncoder()
                    
                    # Store for prediction
                    st.session_state.encoders['strategy'] = encoding_strategy
                    st.session_state.encoders['target_encoder'] = target_encoder
                    st.session_state.encoders['cat_features'] = cat_features
                    st.session_state.encoders['num_features'] = num_features
                    
                    # Fit and transform
                    if cat_features:
                        X_train[cat_features] = target_encoder.fit_transform(X_train[cat_features], y_train)
                        X_test[cat_features] = target_encoder.transform(X_test[cat_features])
                
                else:  # Combination
                    # Use target encoding for high cardinality features and one-hot for low cardinality
                    low_cardinality_features = [f for f in categorical_features if f in selected_features and 
                                             X[f].nunique() < 10]
                    high_cardinality_features = [f for f in categorical_features if f in selected_features and 
                                              f not in low_cardinality_features]
                    num_features = [f for f in numerical_features if f in selected_features]
                    
                    # Create encoders
                    target_encoder = TargetEncoder()
                    
                    # Store for prediction
                    st.session_state.encoders['strategy'] = encoding_strategy
                    st.session_state.encoders['target_encoder'] = target_encoder
                    st.session_state.encoders['low_cardinality'] = low_cardinality_features
                    st.session_state.encoders['high_cardinality'] = high_cardinality_features
                    st.session_state.encoders['num_features'] = num_features
                    
                    # Transform high cardinality features
                    if high_cardinality_features:
                        X_train[high_cardinality_features] = target_encoder.fit_transform(X_train[high_cardinality_features], y_train)
                        X_test[high_cardinality_features] = target_encoder.transform(X_test[high_cardinality_features])
                    
                    # For low cardinality, we'll do one-hot encoding manually
                    if low_cardinality_features:
                        X_train_encoded = pd.get_dummies(X_train, columns=low_cardinality_features, drop_first=True)
                        X_test_encoded = pd.get_dummies(X_test, columns=low_cardinality_features, drop_first=True)
                        
                        # Ensure same columns in test and train
                        for col in X_train_encoded.columns:
                            if col not in X_test_encoded.columns:
                                X_test_encoded[col] = 0
                        X_test_encoded = X_test_encoded[X_train_encoded.columns]
                        
                        X_train, X_test = X_train_encoded, X_test_encoded
                
                # Apply balancing based on selected method
                if balancing:
                    try:
                        from collections import Counter
                        
                        # Count class occurrences
                        class_counts = Counter(y_train)
                        st.info(f"Dataset has {len(class_counts)} unique crime types.")
                        
                        if balancing_method == "SMOTE (for moderately imbalanced data)":
                            from imblearn.over_sampling import SMOTE
                            
                            # Find classes with only one sample
                            single_sample_classes = [cls for cls, count in class_counts.items() if count < 2]
                            
                            if single_sample_classes:
                                # Filter out single-sample classes for SMOTE
                                keep_indices = [i for i, y in enumerate(y_train) if y not in single_sample_classes]
                                X_train_filtered = X_train.iloc[keep_indices] if hasattr(X_train, 'iloc') else X_train[keep_indices]
                                y_train_filtered = y_train[keep_indices]
                                
                                # Apply SMOTE to classes with sufficient samples
                                smote = SMOTE(random_state=42)
                                X_resampled, y_resampled = smote.fit_resample(X_train_filtered, y_train_filtered)
                                
                                # Add back the single-sample classes
                                single_sample_indices = [i for i, y in enumerate(y_train) if y in single_sample_classes]
                                if single_sample_indices:
                                    X_single = X_train.iloc[single_sample_indices] if hasattr(X_train, 'iloc') else X_train[single_sample_indices]
                                    y_single = y_train[single_sample_indices]
                                    
                                    # Combine the datasets
                                    if hasattr(X_resampled, 'iloc'):  # If pandas DataFrame
                                        X_train = pd.concat([X_resampled, X_single])
                                        y_train = np.concatenate([y_resampled, y_single])
                                    else:  # If numpy array
                                        X_train = np.vstack([X_resampled, X_single])
                                        y_train = np.concatenate([y_resampled, y_single])
                                else:
                                    X_train, y_train = X_resampled, y_resampled
                                
                                st.warning(f"Some crime types had only one example and couldn't be balanced with SMOTE.")
                            else:
                                # Standard SMOTE if all classes have enough samples
                                smote = SMOTE(random_state=42)
                                X_train, y_train = smote.fit_resample(X_train, y_train)
                        
                        elif balancing_method == "RandomOverSampler (for highly imbalanced data)":
                            from imblearn.over_sampling import RandomOverSampler
                            
                            # This is safer for highly imbalanced datasets
                            ros = RandomOverSampler(random_state=42)
                            X_train, y_train = ros.fit_resample(X_train, y_train)
                            st.info("Applied RandomOverSampler to balance classes")
                        
                        elif balancing_method == "Class Weights (for highly imbalanced data)":
                            # Calculate class weights
                            class_weights = {c: len(y_train) / (len(class_counts) * count) 
                                            for c, count in class_counts.items()}
                            
                            # Store for model training
                            st.session_state.class_weights = class_weights
                            st.info("Calculated class weights to handle imbalance during training")
                    
                    except Exception as e:
                        st.warning(f"Could not apply {balancing_method}: {str(e)}. Proceeding with original data.")
                
                # Create and train the model with class weights if applicable
                if balancing_method == "Class Weights (for highly imbalanced data)":
                    class_weights = st.session_state.class_weights
                    
                    if algorithm == "Random Forest":
                        model = RandomForestClassifier(
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            min_samples_split=min_samples_split,
                            class_weight=class_weights,
                            random_state=42,
                            n_jobs=-1
                        )
                    elif algorithm == "XGBoost":
                        # For XGBoost, need to handle class weights differently
                        sample_weights = np.ones(len(y_train))
                        for idx, y in enumerate(y_train):
                            sample_weights[idx] = class_weights.get(y, 1.0)
                        
                        # Make sure categorical columns are properly encoded
                        X_train_xgb = X_train.copy()
                        
                        # Convert object columns to category or numeric
                        for col in X_train_xgb.columns:
                            if X_train_xgb[col].dtype == 'object':
                                # Convert to category type
                                X_train_xgb[col] = X_train_xgb[col].astype('category')
                        
                        model = xgb.XGBClassifier(
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            learning_rate=learning_rate,
                            random_state=42,
                            n_jobs=-1,
                            enable_categorical=True  # Enable categorical feature support
                        )
                        
                        # Will use sample_weights during fitting
                        st.session_state.sample_weights = sample_weights
                        
                        # Store the transformed dataframe
                        st.session_state.X_train_xgb = X_train_xgb
                        
                    else:  # LightGBM
                        model = lgb.LGBMClassifier(
                            n_estimators=n_estimators,
                            num_leaves=num_leaves,
                            learning_rate=learning_rate,
                            class_weight=class_weights,
                            random_state=42,
                            n_jobs=-1
                        )
                else:
                    # Regular models without class weights
                    if algorithm == "Random Forest":
                        model = RandomForestClassifier(
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            min_samples_split=min_samples_split,
                            random_state=42,
                            n_jobs=-1
                        )
                    elif algorithm == "XGBoost":
                        model = xgb.XGBClassifier(
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            learning_rate=learning_rate,
                            random_state=42,
                            n_jobs=-1
                        )
                    else:  # LightGBM
                        model = lgb.LGBMClassifier(
                            n_estimators=n_estimators,
                            num_leaves=num_leaves,
                            learning_rate=learning_rate,
                            random_state=42,
                            n_jobs=-1
                        )
                
                # Train the model (with sample weights if applicable)
                logger.info(f"Training {algorithm} model with {len(X_train)} samples")
                logger.info(f"Features: {X_train.columns.tolist()}")
                logger.info(f"Target distribution: {np.unique(y_train, return_counts=True)}")
                
                try:
                    if algorithm == "XGBoost" and balancing_method == "Class Weights (for highly imbalanced data)":
                        # Use the properly formatted data for XGBoost
                        X_train_xgb = st.session_state.X_train_xgb
                        model.fit(X_train_xgb, y_train, sample_weight=st.session_state.sample_weights)
                    elif algorithm == "XGBoost":
                        # For XGBoost without class weights, still need properly formatted data
                        X_train_xgb = X_train.copy()
                        for col in X_train_xgb.columns:
                            if X_train_xgb[col].dtype == 'object':
                                X_train_xgb[col] = X_train_xgb[col].astype('category')
                        
                        model.fit(X_train_xgb, y_train)
                        # Store for prediction
                        st.session_state.X_train_xgb = X_train_xgb
                    else:
                        model.fit(X_train, y_train)
                    logger.info("Model training completed successfully")
                except Exception as e:
                    logger.error(f"Error during model training: {str(e)}")
                    st.error(f"Error during model training: {str(e)}")
                    # Don't use return here as it's outside a function
                
                # Store the model
                st.session_state.model = model
                
                # Evaluate the model
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
                
                # Store metrics
                st.session_state.model_metrics = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1
                }
                
                # Display results
                st.success(f"Model trained successfully!")
                st.subheader("Model Performance")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Accuracy", f"{accuracy:.3f}")
                col2.metric("Precision", f"{precision:.3f}")
                col3.metric("Recall", f"{recall:.3f}")
                col4.metric("F1 Score", f"{f1:.3f}")
                
                # Display confusion matrix for top classes
                st.subheader("Confusion Matrix (Top 5 Classes)")
                y_test_names = label_encoder.inverse_transform(y_test)
                y_pred_names = label_encoder.inverse_transform(y_pred)
                
                # Get top 5 classes by frequency
                top_classes = pd.Series(y_test_names).value_counts().head(5).index.tolist()
                
                # Filter to only include top classes
                mask_test = np.isin(y_test_names, top_classes)
                mask_pred = np.isin(y_pred_names, top_classes)
                
                if len(top_classes) > 1:  # Only create confusion matrix if we have multiple classes
                    cm = confusion_matrix(
                        y_test_names[mask_test], 
                        y_pred_names[mask_test],
                        labels=top_classes
                    )
                    
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=top_classes, yticklabels=top_classes)
                    plt.ylabel('Actual')
                    plt.xlabel('Predicted')
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                # Feature importance
                if hasattr(model, 'feature_importances_'):
                    st.subheader("Feature Importance")
                    importances = model.feature_importances_
                    feature_names = X_train.columns
                    
                    # Sort by importance
                    indices = np.argsort(importances)[::-1]
                    
                    # Plot top 15 features
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.barplot(
                        x=importances[indices][:15], 
                        y=[feature_names[i] for i in indices][:15],
                        palette="viridis"
                    )
                    plt.title('Feature Importances')
                    plt.tight_layout()
                    st.pyplot(fig)

elif page == "Prediction":
    st.title("Crime Prediction")
    
    if st.session_state.model is None:
        st.warning("Please train a model first in the 'Model Training' tab.")
    else:
        st.success("Model is ready for predictions!")
        
        # Display model metrics
        if st.session_state.model_metrics:
            st.subheader("Current Model Performance")
            metrics = st.session_state.model_metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
            col2.metric("Precision", f"{metrics['precision']:.3f}")
            col3.metric("Recall", f"{metrics['recall']:.3f}")
            col4.metric("F1 Score", f"{metrics['f1']:.3f}")
        
        # Get the processed data for reference values
        processed_data = preprocess_data(data, for_training=False)
        
        # Prediction form
        st.subheader("Enter Details to Predict Crime Type")
        with st.form("crime_prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                age_input = st.number_input("Offender Age", min_value=10, max_value=100, value=30)
                gender_display = st.selectbox("Gender", ['Male', 'Female'])
                gender_input = 'M' if gender_display == 'Male' else 'F'
                district_input = st.selectbox("District", st.session_state.district_options)
                neighborhood_input = st.selectbox("Neighborhood", st.session_state.neighborhood_options)
            
            with col2:
                hour_input = st.number_input("Hour of Arrest (0-23)", min_value=0, max_value=23, value=12)
                day_of_week_input = st.number_input("Day of Week (0=Mon, 6=Sun)", min_value=0, max_value=6, value=0)
                month_input = st.number_input("Month", min_value=1, max_value=12, value=1)
                year_input = st.number_input("Year", min_value=2010, max_value=2025, value=2023)
            
            # Calculate derived features
            time_of_day_input = pd.cut([hour_input], bins=[0, 6, 12, 18, 24], labels=['Night', 'Morning', 'Afternoon', 'Evening'], right=False)[0]
            weekend_input = 1 if day_of_week_input >= 5 else 0
            
            # Interaction features
            age_timeofday_input = f"{age_input}_{time_of_day_input}"
            district_timeofday_input = f"{district_input}_{time_of_day_input}"
            
            # Age category
            age_category = pd.cut([age_input], bins=[0, 18, 30, 50, 100], labels=['Minor', 'Young', 'Middle', 'Senior'])[0]
            gender_age_input = f"{gender_input}_{age_category}"
            
            submitted = st.form_submit_button("Predict")
            
            if submitted:
                # Prepare input data with all possible features
                input_data = pd.DataFrame({
                    'Age': [age_input],
                    'Gender': [gender_input],
                    'District': [district_input],
                    'Neighborhood': [neighborhood_input],
                    'Hour': [hour_input],
                    'DayOfWeek': [day_of_week_input],
                    'Month': [month_input],
                    'Year': [year_input],
                    'TimeOfDay': [time_of_day_input],
                    'Weekend': [weekend_input],
                    'Age_TimeOfDay': [age_timeofday_input],
                    'District_TimeOfDay': [district_timeofday_input],
                    'Gender_Age': [gender_age_input]
                })
                
                # Filter features based on what was used in training
                input_data = input_data[st.session_state.feature_names]
                
                # Apply encodings based on strategy
                encoding_strategy = st.session_state.encoders['strategy']
                
                if encoding_strategy == "Target Encoding":
                    cat_features = st.session_state.encoders['cat_features']
                    target_encoder = st.session_state.encoders['target_encoder']
                    
                    if cat_features:
                        input_data[cat_features] = target_encoder.transform(input_data[cat_features])
                
                elif encoding_strategy == "Combination":
                    high_cardinality_features = st.session_state.encoders.get('high_cardinality', [])
                    low_cardinality_features = st.session_state.encoders.get('low_cardinality', [])
                    target_encoder = st.session_state.encoders.get('target_encoder')
                    
                    # Apply target encoding to high cardinality features
                    if high_cardinality_features and target_encoder:
                        input_data[high_cardinality_features] = target_encoder.transform(input_data[high_cardinality_features])
                    
                    # One-hot encode low cardinality features
                    if low_cardinality_features:
                        input_data = pd.get_dummies(input_data, columns=low_cardinality_features, drop_first=True)
                        
                        # Handle missing columns
                        model_features = st.session_state.model.feature_names_in_
                        for feature in model_features:
                            if feature not in input_data.columns:
                                input_data[feature] = 0
                        
                        # Ensure correct order of columns
                        input_data = input_data[model_features]
                
                # Make prediction
                if 'model' in st.session_state and isinstance(st.session_state.model, xgb.XGBClassifier):
                    # For XGBoost, we need to handle categorical features properly
                    input_data_xgb = input_data.copy()
                    for col in input_data_xgb.columns:
                        if input_data_xgb[col].dtype == 'object':
                            input_data_xgb[col] = input_data_xgb[col].astype('category')
                    
                    prediction_encoded = st.session_state.model.predict(input_data_xgb)[0]
                else:
                    prediction_encoded = st.session_state.model.predict(input_data)[0]
                
                prediction_label = st.session_state.label_encoder.inverse_transform([prediction_encoded])[0]
                
                # Calculate prediction probabilities
                if 'model' in st.session_state and isinstance(st.session_state.model, xgb.XGBClassifier):
                    prediction_proba = st.session_state.model.predict_proba(input_data_xgb)[0]
                else:
                    prediction_proba = st.session_state.model.predict_proba(input_data)[0]
                
                # Display the result
                st.success(f"Predicted Crime Type: {prediction_label}")
                
                # Show top 3 most likely crime types with probabilities
                st.subheader("Top 3 Most Likely Crime Types:")
                
                # Get the indices of the top 3 probabilities
                top_indices = prediction_proba.argsort()[-3:][::-1]
                top_classes = st.session_state.label_encoder.inverse_transform(top_indices)
                top_probas = prediction_proba[top_indices]
                
                # Create a DataFrame for visualization
                top_predictions = pd.DataFrame({
                    'Crime Type': top_classes,
                    'Probability': top_probas
                })
                
                # Display as chart
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(data=top_predictions, x='Probability', y='Crime Type', palette='viridis')
                plt.xlim(0, 1)
                plt.tight_layout()
                st.pyplot(fig)
                
                # Feature importance for this prediction (if SHAP is available)
                st.subheader("Note on Explainability")
                st.info("For additional insights, consider implementing SHAP values to explain this specific prediction.")