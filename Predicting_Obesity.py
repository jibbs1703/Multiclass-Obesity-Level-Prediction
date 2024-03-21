# Predicting Obesity Levels Using the Random Forest Classifier Model
# Import Necessary Libraries 
# Data Exploration, Analysis and Visualization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Data Splitting, Dealing with Labels/Categories and Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Import Data and Check Shape (Please replace filepath to suit your needs)
filepath = 'C:/Users/New/ObesityData.csv'
obesity = pd.read_csv(filepath)
print(obesity.shape)

#Check Datatypes Present and if any Null Values are Present (using .info() or .dtypes)
obesity.info()

# Transform Column Names by Renaming and Making all Variables Uppercase for Consistency
obesity.rename(columns={'Gender' : 'Gender'.upper(), 'Age': 'Age'.upper(), 'Height':'Height'.upper(), 'Weight' : 'Weight'.upper(),'family_history_with_overweight': 'HISTORY', 'NObeyesdad': 'OBLEVEL'}, inplace=True)
print(obesity.columns)

# Feature Engineering Based on the Data Dictionary
# First Convert Float Datatypes into Int, Except Height and Weight that are actually Floats according to the Data Dictionary
ignore_floats = ['HEIGHT', 'WEIGHT']
for col in obesity.drop(columns=ignore_floats, axis=1).columns:
    if obesity[col].dtypes == 'float64':
         obesity[col] = obesity[col].astype(int)
            
# Convert the Integers for the Transformed Variables except AGE into Categories According to the Data Dictionary
# Create Dictionaries to Map the Variable Elements
cat_fvcv = {1:"Never", 2: "Sometimes", 3 :"Always"}
cat_ch2o = {1: "Less_than_1L", 2: "Between_1_and_2L", 3: "More_than_2L"}
cat_faf = {0: "I_do_not", 1: "One_or_Two_Days", 2: "Two_or_Four_Days", 3: "Four_or_Five_Days"}
cat_tue = {0: "Zero_to_Two_Hours", 1: "Between_Three_and_Five_Hours", 2: "More_than_Five_Hours" }

# Map Variable Elements Based on Dictionary Created
obesity["FCVC"] = obesity["FCVC"].map(cat_fvcv)
obesity["CH2O"] = obesity["CH2O"].map(cat_ch2o)
obesity["FAF"] = obesity["FAF"].map(cat_faf)
obesity["TUE"] = obesity["TUE"].map(cat_tue)

# View Description of Categorical Attributes and Target Variable
obesity.describe(exclude=[np.number])

# View Description of Numeric Attributes 
obesity.describe()

# Data Exploration and Vizualization
# Sort the Variables into Separate Lists of the Categorical and Numerical Columns in the Dataset
num_cols = list(obesity.select_dtypes(exclude=['object']).columns)
print(num_cols)
cat_cols = list(obesity.select_dtypes(include=['object']).columns)
print(cat_cols)

# Use BarPlots to View the Distribution of the Numerical Variables
plt.figure(figsize=(15, 10))
for i, col in enumerate(num_cols, 1):
    plt.subplot(2, 2, i)
    sns.histplot(obesity[col], kde=True)
    plt.title(col)
plt.tight_layout()
plt.show()

# Use Countplots to View the Distribution of the Categorical Variables
plt.figure(figsize=(15, 20))
for i, col in enumerate(cat_cols, 1):
    if col != "OBLEVEL":
        plt.subplot(6, 2, i)
        sns.countplot(data=obesity, x=col)
        plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# Use a Countplot to View the Distribution of the Target Variable - Obesity Level
plt.figure(figsize=(12, 6))
sns.countplot(data=obesity, x='OBLEVEL')
plt.title('Distribution of Obesity Level Among Respondents in Dataset')
plt.xlabel('Obesity Level')
plt.ylabel('Frequency')
plt.show()

# Instantiate Label Encoder and Label Encode OBLEVEL - the Target Variable
label_encoder = LabelEncoder()
obesity['OBLEVEL'] = label_encoder.fit_transform(obesity['OBLEVEL'])
obesity = pd.get_dummies(obesity)

# Create Target Variables and Feature Variables from Obesity Dataset
X = obesity.drop(['OBLEVEL'], axis=1)
y = obesity['OBLEVEL']

# Split Data into Train and Test Portions and A Third Category Called Validation (For Decision Trees and Random Forests)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Train Model and Check Accuracy of Training Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
train_preds = model.predict(X_train)
train_accuracy = accuracy_score(y_train, train_preds )
print(f"Training Accuracy: {train_accuracy}")

# Check Accuracy of Model on Validation Data 
val_preds = model.predict(X_val)
val_accuracy = accuracy_score(y_val, val_preds)
print(f"Validation Accuracy: {val_accuracy}")

# Check Model Performance with Test Data
test_preds = model.predict(X_test)
test_accuracy = accuracy_score(y_test, test_preds)
print(f"Test Accuracy: {test_accuracy}")

# Create a Feature Importance Series to Visualize the Most and Least Important Predictors of Obesity Levels
features = X_train.columns
importances = model.feature_importances_
feat_imp = pd.Series(importances, index = features).sort_values()

# 15 Most Important Predictors of Obesity Levels
feat_imp.tail(15).plot(kind = 'barh')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.title("15 Most Important Features in Predicting Obesity Levels");

# 15 Least Important Predictors of Obesity Levels
feat_imp.head(15).plot(kind = 'barh')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.title("15 Least Important Features in Predicting Obesity Levels");

# Confusion matrix for Validation Data
conf_matrix = pd.crosstab(y_val, val_preds, rownames=['Actual'], colnames=['Predicted'])
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.show()

# Confusion matrix for Test Data
conf_matrix = pd.crosstab(y_test, test_preds, rownames=['Actual'], colnames=['Predicted'])
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Oranges')
plt.title('Confusion Matrix')
plt.show()

# Export Results to a DataFrame (Predicted vs Actual)
results = pd.DataFrame({'Predicted': test_preds, 'Actual': y_test})

# Create Target Map Dictionary to View Data as Objects and Not Integers
obmap  ={0:'Insufficient_Weight',1:'Normal_Weight',2:'Overweight_Level_I',3:'Overweight_Level_II',4:'Obesity_Type_I',5:'Obesity_Type_II',6:'Obesity_Type_III'}
results = results.apply(lambda col: col.map(obmap))
results.head()

#Comparisms of the Actual and Predicted Obesity Levels (as Proportions)
print(results['Predicted'].value_counts(normalize= True))
print(results['Actual'].value_counts(normalize= True))

# Save Model as .pkl file
pickle.dump(model, open('model.pkl', 'wb'))
