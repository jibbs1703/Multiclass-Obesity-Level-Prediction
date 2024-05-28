# Import Necessary Libraries

# Data Analysis and Visualization Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

# Machine Learning Libraries
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Deep Learning Library and Modules
import torch
import torch.nn as nn
import torch.nn.functional as fun
from torch.utils.data import TensorDataset, DataLoader



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
sns.countplot(data=train, x='NObeyesdad')
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
    cat_fvcv = {1: "Never", 2: "Sometimes", 3: "Always"}
    cat_ch2o = {1: "Less_than_1L", 2: "Between_1_and_2L", 3: "More_than_2L"}
    cat_faf = {0: "I_do_not", 1: "One_or_Two_Days", 2: "Two_or_Four_Days", 3: "Four_or_Five_Days"}
    cat_tue = {0: "Zero_to_Two_Hours", 1: "Between_Three_and_Five_Hours", 2: "More_than_Five_Hours"}

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

    df = df.drop(columns=['id'])

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
X_train = trr.drop(target, axis=1)
y_train = trr[target]
X_test = tee

# Create Simple Deep Learning Prediction Model (2 Hidden Layers)
class Model(nn.Module):

    def __init__(self, in_features=18, h1=72, h2=144, out_features=7):
        super(Model, self).__init__()  # Instantiate the nn.Module
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)

    def forward(self, X):
        X = fun.relu(self.fc1(X))
        X = fun.relu(self.fc2(X))
        X = self.out(X)

        return X

# Set Manual seed for Randomization
torch.manual_seed(41)

# Create an instance of model
model = Model()

# Convert DF to Numpy Array
X_train = X_train.values
X_test = X_test.values

y_train = y_train.values

# Feature Neurons are Set to Floats
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)

# LongTensor Used Here Because it is Categorical and not Continuous
y_train = torch.LongTensor(y_train)

# Set Criterion for Error Measurment
criterion = nn.CrossEntropyLoss()

# Set Optimizer as Adam and Set Learning Rate
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

epochs = 1000
losses = []
for epoch in range(epochs):
    y_pred = model.forward(X_train)
    loss = criterion(y_pred, y_train)
    losses.append(loss.detach().numpy())
    if epoch % 500 == 0:
        print(f" Epoch: {epoch}, Loss: {loss}")

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

plt.plot(range(epochs), losses);

result = []
with torch.no_grad():
    for i, data in enumerate(X_test):
        y_val = model.forward(data)
        result.append(y_val.argmax().item())

le = LabelEncoder()
le.fit(train_id['NObeyesdad'])
result = le.inverse_transform(result)

# Send Predictions to a Dataframe
results = pd.DataFrame({'id': test_id['id'], 'NObeyesdad': result})

# Save Submission to Output File For Submission
results.to_csv('C:/Users/New/GitProjects/MyProjects/Predicting-Obesity-Levels/project_files/submission_pytorch.csv', index=False)
print("Submission File was Saved")
