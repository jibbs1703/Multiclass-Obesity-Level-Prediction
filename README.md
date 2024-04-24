# Multiclass Obesity Prediction
## Overview
This repository contains models that predict the obesity level of patients based on their eating/lifestyle habits and physical condition. The target variable here is a multi-class variable with seven levels - Insufficient Weight, Normal Weight, Overweight Level I, Overweight Level II, Obesity Type I, Obesity Type II and Obesity Type III. 

## Dataset For Analysis
The novel version of the dataset used for this project can be found at [UCI Machine Learning Repository](https://archive-beta.ics.uci.edu/dataset/544). The synthetically expanded version can be found on [Kaggle](https://www.kaggle.com/competitions/playground-series-s4e2/overview) as a competition dataset. The dataset features are the same from both sources but [Kaggle](https://www.kaggle.com/competitions/playground-series-s4e2/overview) offers a means of checking the model perfomance on a larger data sample. The original dataset had 2111 observations while the synthetically generated version of the dataset had 34598 observations. 

The data collected covered  demographic data as well as data on eating habits and physical condition from individuals from Colombia, Peru and Mexico. The data contains 16 features and the target variable.The feature attributes related to eating habits are Frequent consumption of high caloric food , Frequency of consumption of vegetables, Number of main meals, Consumption of food between meals, Consumption of water daily, and Consumption of alcohol. The feature attributes related to physical condition are Calories consumption monitoring, Physical activity frequency, Time using technology devices, Transportation used. Demographic attributes such as Gender, Age, Height and Weight were also recorded.

## Python Packages and Modules Needed
[Pandas](https://pandas.pydata.org/) and [NumPy](https://numpy.org/) are imported for Data Manipulation and Wrangling. [Seaborn](https://seaborn.pydata.org/) and [Matplotlib](https://matplotlib.org/) are employed for the visualuzations. The Machine Learning Models and accessory methods are imported from [Scikit-Learn](https://scikit-learn.org/) and [XGBoost](https://xgboost.readthedocs.io/en/stable/#).

## Chronology 
- Import necessary libraries and datasets.
- Identify datatypes present and make appropriate conversions.
- Preprocess dataset - deal with missing values scale numeric features and label categorical features accordingly.
- Split data into the features and target.
- Split data into training and test datasets (not necessary for [Kaggle](https://www.kaggle.com/competitions/playground-series-s4e2/overview) dataset).
- Training machine learning models on training dataset and check training accuracy.
- Tune Hyperparameters of the model (if necessary). 
- Use trained model on test dataset and make prediction.
- Save predictions to a desired file format.

## Model Results
**Decision Tree Classifier** : The Decision Tree Classifier Model attained a 89.82% accuracy on the Training Dataset and 86.77% accuracy score on Test Dataset. 

**XGBoost Classifier** : The XGBoost Classifier Model attained a 90.5% accuracy on the Training Dataset and 88.18% accuracy score on Test Dataset. 

## Author(s)
- **Abraham Ajibade** [Linkedin](https://www.linkedin.com/in/abraham-ajibade-759772117)
- **Boluwtife Olayinka** [Linkedin](https://www.linkedin.com/in/ajibade-bolu/) 
