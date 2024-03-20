# Loan Status Prediction

## Problem Overview
You are tasked with developing a machine learning model to predict the loan status of applicants based on various features. The goal is to classify whether a loan application will be 'Fully Paid' or 'Charged Off', given certain characteristics of the applicant and the loan.

## Objective
The objective of this project is to build a classification model that accurately predicts the loan status of future applicants based on their attributes. The model should be trained on historical data to learn patterns and relationships between the features and the loan status. Once trained, the model should be capable of making predictions on new, unseen data.

## Project Structure

1. **Data Collection and Preprocessing:**
   - Collected data from 'credit_train.csv' and 'credit_test.csv' datasets.
   - Removed 'Loan ID' and 'Customer ID' columns.
   - Handled duplicate rows and missing values using appropriate strategies (e.g., mean imputation, dropping rows).
   - Encoded categorical features and standardized numeric features for modeling.

2. **Exploratory Data Analysis (EDA):**
   - Conducted EDA to understand data distributions, identify outliers, and visualize feature relationships.
   - Utilized box plots, histograms, and correlation matrices for data exploration.

3. **Feature Engineering:**
   - Encoded categorical variables using OrdinalEncoder.
   - Scaled numerical features using StandardScaler for model compatibility.

4. **Model Building and Selection:**
   - Implemented Logistic Regression, Decision Tree Classifier, and Random Forest Classifier models.
   - Tuned hyperparameters using GridSearchCV and RandomizedSearchCV for model optimization.

5. **Model Evaluation:**
   - Evaluated model performance using accuracy score, confusion matrix, precision, recall, and F1 score.
   - Selected the Random Forest Classifier model as the best performing model based on evaluation metrics.

6. **Skills and Tools:**
   - Utilized Python programming language.
   - Leveraged libraries such as pandas, numpy, matplotlib, seaborn, and scikit-learn.
   - Applied data preprocessing techniques, EDA, feature engineering, and machine learning algorithms.

