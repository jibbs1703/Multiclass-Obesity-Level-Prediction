# Import Necessary Libraries

# Data Analysis and Visualization Libraries
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

# Machine Learning Libraries
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Import Datasets and Perform General Check on Datasets

# Import Datset from Filepath
train_path = 'C:/Users/New/GitProjects/MyProjects/Predicting-Obesity-Levels/project_files/train.csv'
test_path = 'C:/Users/New/GitProjects/MyProjects/Predicting-Obesity-Levels/project_files/test.csv'

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

print(f"The train dataset has {train.shape[0]} rows and {train.shape[1]} columns")
print(f"The test dataset has {test.shape[0]} rows and {test.shape[1]} columns")

# Check Datasets for Missing Values 
train.info()
test.info()

# Simple Visualizations of Dataset Features

# Visualize the Distribution of the Target in Train Dataset
plt.figure(figsize=(16, 6)) 
sns.countplot(data = train, x = 'NObeyesdad')
plt.ylabel('Frequency');

# Correlation of Numeric Features
num_cols = list(train.select_dtypes(exclude=['object']).columns)
corr_matrix = train[num_cols].corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='viridis')
plt.title('Correlation Matrix of Numerical Features')
plt.show()

# Feature Engineering and Data Pre-Processing

# Create cat_bmi Function to Make BMI Categories
def cat_bmi(bmi):
    if bmi < 18.5:
        return "Underweight"
    elif 18.5 <= bmi < 25:
        return "Healthy weight"
    elif 25 <= bmi < 30:
        return "Overweight"
    elif bmi >= 30:
        return "Obese"

# Create Feature Wrangling Function
def feat_trans(df):
    
    # Create New BMI and BMI Category from Height and Weight
    df['BMI'] = df['Weight'] / (df['Height'] ** 2)
    df['BMI_Cat'] = df['BMI'].apply(cat_bmi) 
    
    # Feature Engineering Based on the Data Dictionary
    # First Convert Float Datatypes into Int, Except Height and Weight that are actually Floats according to the Data Dictionary
    ignore_floats = ['Height', 'Weight']
    for col in df.drop(columns=ignore_floats, axis=1).columns:
        if df[col].dtypes == 'float64':
             df[col] = df[col].astype(int)

    # Convert the Integers for the Transformed Variables except AGE into Categories According to the Data Dictionary
    # Create Dictionaries to Map the Variable Elements
    cat_fvcv = {1:"Never", 2: "Sometimes", 3 :"Always"}
    cat_ch2o = {1: "Less_than_1L", 2: "Between_1_and_2L", 3: "More_than_2L"}
    cat_faf = {0: "I_do_not", 1: "One_or_Two_Days", 2: "Two_or_Four_Days", 3: "Four_or_Five_Days"}
    cat_tue = {0: "Zero_to_Two_Hours", 1: "Between_Three_and_Five_Hours", 2: "More_than_Five_Hours" }

    # Map Variable Elements Based on Dictionary Created
    df["FCVC"] = df["FCVC"].map(cat_fvcv)
    df["CH2O"] = df["CH2O"].map(cat_ch2o)
    df["FAF"] = df["FAF"].map(cat_faf)
    df["TUE"] = df["TUE"].map(cat_tue)
    
    # Scale the Numeric Features
    scaler = StandardScaler()
    num_cols = list(df.select_dtypes(exclude=['object']).columns)
    df[num_cols] = scaler.fit_transform(df[num_cols])
    
    # Encode Categorical Variables In Dataset
    cat_cols = list(df.select_dtypes(include=['object']).columns)
    le = LabelEncoder()
    for col in cat_cols:
        le.fit(df[col])
        df[col] = le.transform(df[col])
        
    df = df.drop(columns = ['id'])
    
    return df

# # Create Copy of Test and Train Datasets Before Feature Transformation
train1 = train
test1 = test
test_id = test.copy()
train_id = train.copy()

# Transform Test and Train Using Transformation Function
tee = feat_trans(test1)
trr = feat_trans(train1)

# Split Datasets into Target and Feature Variables
target = 'NObeyesdad'
X_train = trr.drop(target, axis = 1)
y_train = trr[target]
X_test = tee

# Instantiate XGBoost Model and Create HyperParameter Grid
model = XGBClassifier(random_state = 420)
param_grid = {
    
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5],
    'n_estimators': [50, 100, 200],
    'subsample': [0.5, 0.7, 1],
}

# Train Model and Obtain Best Set of Hyperparameters
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
print(f'Best Set of Hyperparameters: {best_params}')

# Cross Validate the Trained Model
best_model = grid_search.best_estimator_
cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='accuracy')
print(f'Cross-Validation Scores: {cv_scores}')
print(f'Mean CV Accuracy: {cv_scores.mean()}')

# Use Trained Model on Test Dataset and Reverse Label Predictions
predictions = best_model.predict(X_test)
le = LabelEncoder()
le.fit(train_id['NObeyesdad'])
predictions = le.inverse_transform(predictions)

# Export Prediction to Dataframe 
results = pd.DataFrame({'id': test_id['id'], 'NObeyesdad': predictions})

# Save Submission to Output File For Submission
results.to_csv('C:/Users/New/GitProjects/MyProjects/Predicting-Obesity-Levels/project_files/submission_xgboost.csv', index=False)
print("submission_xgboost was saved")