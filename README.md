# CHD Prediction Using Machine Learning
## Coronary heart disease Prediction Using Decision Tree Classifier
This project aims to build a Decision Tree Classifier model to predict the occurrence of Coronary Heart Disease (CHD) based on the input features. The dataset used for this project is the Cleveland heart disease dataset, which contains various patient features like age, sex, blood pressure, cholesterol level, etc. along with their target CHD status.
### Introduction 
The purpose of this notebook is to predict the risk of Coronary Heart Disease (CHD) based on various risk factors. CHD is a condition that occurs when the arteries that supply blood to the heart become blocked, leading to heart attacks and other serious complications.

To predict the risk of CHD, we will use a decision tree classifier model. This model will be trained on a dataset of patient information, including demographic, lifestyle, and medical data.

### Data Exploration
The first step in building our CHD prediction model is to explore the dataset and gain an understanding of the variables involved. We use various data visualization techniques such as histograms, box plots, and scatter plots to identify potential patterns and correlations in the data.

### Data Preprocessing
The dataset contains missing values and categorical variables that need to be handled before training our model. In the preprocessing step, we handle the missing values using the mean or median of the respective column. We also convert the categorical variables into numerical format using one-hot encoding.

### Model Building 
In this step, we build a decision tree classifier model using scikit-learn. We split the dataset into a training set and a test set, and then train the model on the training set. We use various hyperparameters such as max_depth and min_samples_split to tune the model for optimal performance.

### Model Evaluation
Finally, we evaluate the performance of our model on the test set. We use metrics such as accuracy, precision, recall, and F1 score to evaluate the model's performance. Additionally, we create a confusion matrix to visualize the model's performance in predicting true positives, true negatives, false positives, and false negatives.

### Conclusion
In this notebook, we have demonstrated the use of a decision tree classifier model to predict the risk of CHD. We have explored the dataset, preprocessed the data, built a model, and evaluated its performance. The notebook serves as an example of how machine learning techniques can be used to solve real-world problems in healthcare.

### Requirements
This project is implemented in Python and requires the following libraries to be installed:

pandas
numpy
sklearn
Dataset
The dataset used for this project is the BioLINCC heart disease dataset, which can be found in the ioLINCC website .
### Implementation

The project consists of the following files:

coronary_heart_disease_prediction.ipynb: a Jupyter notebook containing the implementation of the project.
heart.csv: the dataset used for the project.
README.md: a readme file describing the project.
Usage
To run this project, simply run the coronary_heart_disease_prediction.ipynb file in a Jupyter notebook environment.

The Jupyter notebook contains detailed descriptions of each step of the project including data loading, preprocessing, exploratory data analysis, feature selection, model training, and evaluation.

### Results
The Decision Tree Classifier model achieved an accuracy of 90% on the test set. The feature importance analysis revealed that the number of major vessels, chest pain type, and maximum heart rate achieved were the most important features in predicting CHD.

### Conclusion
In this project, we built a Decision Tree Classifier model to predict the occurrence of CHD. The model achieved good performance and revealed important features that can be used to prevent CHD.
