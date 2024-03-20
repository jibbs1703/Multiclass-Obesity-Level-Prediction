# Predicting-Obesity-Levels
This repository contains the predictions of obesity levels of patients based on their eating habits and physical condition. The target variable, the obesity level, is a 7-level target variable, making this project different from a binary classification model. To make the predictions on obesity levels, a Random Forest Classifier Model is used.

## Dataset For Analysis
The dataset used for this project was obtained from [UCI Machine Learning Repository](https://archive-beta.ics.uci.edu/dataset/544). The data collected covered  demographic data as well as data on eating habits and physical condition from individuals from Colombia, Peru and Mexico. 

The data contains 17 attributes and 2111 records. The target attribute is the Obesity Level (OBLEVEL), that allows classification of the data using the values of Insufficient Weight, Normal Weight, Overweight Level I, Overweight Level II, Obesity Type I, Obesity Type II and Obesity Type III (seven levels). The feature attributes related to eating habits are Frequent consumption of high caloric food (FAVC), Frequency of consumption of vegetables (FCVC), Number of main meals (NCP), Consumption of food between meals (CAEC), Consumption of water daily (CH20), and Consumption of alcohol (CALC). The feature attributes related to physical condition are Calories consumption monitoring (SCC), Physical activity frequency (FAF), Time using technology devices (TUE), Transportation used (MTRANS). Demographic attributes such as Gender (GENDER), Age (AGE), Height (HEIGHT) and Weight (WEIGHT) were also recorded.

## Python Packages and Modules Needed
- [Pandas](https://pandas.pydata.org/)
- [NumPy](https://numpy.org/)
- [Seaborn](https://seaborn.pydata.org/)
- [Matplotlib](https://matplotlib.org/)
- [Scikit-Learn](https://scikit-learn.org/)
- [Pickle](Lib/pickle.py)

## Description of Steps in Code Script
The most important part of this project was ensuring all features were in their actual data types. Hence, the source of the data was traced to elicit the true datatypes for all features in the dataset which were in the wrong formats, mainly as numeric data, when they should have been categorical. Based on the data dictionary, the variables Age (AGE) and Number of main meals (NCP)  were transformed into the integer datatype, while Frequency of consumption of vegetables (FCVC), Consumption of water daily (CH20), Physical activity frequency (FAF) and Time using technology devices (TUE) were converted into integers before being transformed into the appropriate categories, according to the data dictionary. Proper data cleaning and feature engineering is key to the success of this model as inappropriate datatypes would simply carry on into the model undectected. 

After attribute transformation and feature engineering was completed, the numeric and categorical attributes described statistically and visualized using barplots and histograms. The visualzations showed how the features were distributed. For the target attribute, the visualizations helped show that there was no significant misbalance in the distribution of the levels in the target, meaning rebalancing of the target was not needed for this project. The data was also checked for missing variables earlier in the analysis and 
since no mising values were present, the model used had no use for imputing missing values and value imputer methods were not imported for this model.

The dataset was split into training, validation and test portions in 80:20 splits and the Random Forest Classifier Model was used to make predictions of obesity levels, based on the features in the data. 

## Model Results
The model was first trained and validated with minimal hyperparameter tuning, giving accuracy scores of 100% for the training data and 93% for the validation data. Subsequently, the trained model had an accuracy score of 92.20% on the test data. The results indicate that the model was able to correctly predict 92% of the target variable - the obesity levels. The table below shows the proportions of the levels predicted obesity levels compared to the actual obesity levels observed. 

| Obesity Level| Actual Proportion | Predicted Proportion  |
|  :------:    |  :---------:      |      :---------:      |
| Overweight_Level_I | 0.184397      |  0.177305| 
| Obesity_Type_I  |  0.148936      |     0.148936|
| Normal_Weight   |  0.146572      | 0.141844|
| Overweight_Level_II |  0.137116      |    0.139480|
| Insufficient_Weight | 0.132388      | 0.134752 |
| Obesity_Type_II  |  0.132388     |  0.134752    |
| Obesity_Type_III | 0.118203      | 0.122931 |
      

## Author(s)
- **Abraham Ajibade** [Linkedin](https://www.linkedin.com/in/abraham-ajibade-759772117)
- **Boluwtife Olayinka** [Linkedin](https://www.linkedin.com/in/ajibade-bolu/) 
