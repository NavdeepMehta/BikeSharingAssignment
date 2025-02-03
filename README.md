# Bike Sharing Assignment (BoomBikes)

## **Problem Statement**

A bike-sharing system is a service in which bikes are made available for shared use to individuals on a short term basis for a price or free. Many bike share systems allow people to borrow a bike from a "dock" which is usually computer-controlled wherein the user enters the payment information, and the system unlocks it. This bike can then be returned to another dock belonging to the same system.

A US bike-sharing provider **BoomBikes** has recently suffered considerable dips in their revenues due to the ongoing Corona pandemic. The company is finding it very difficult to sustain in the current market scenario. So, it has decided to come up with a mindful business plan to be able to accelerate its revenue as soon as the ongoing lockdown comes to an end, and the economy restores to a healthy state.

In such an attempt, BoomBikes aspires to understand the demand for shared bikes among the people after this ongoing quarantine situation ends across the nation due to Covid-19. They have planned this to prepare themselves to cater to the people's needs once the situation gets better all around and stand out from other service providers and make huge profits.

They have contracted a consulting company to understand the factors on which the demand for these shared bikes depends. Specifically, they want to understand the factors affecting the demand for these shared bikes in the American market. The company wants to know:

- Which variables are significant in predicting the demand for shared bikes.
- How well those variables describe the bike demands <br>

Based on various meteorological surveys and people's styles, the service provider firm has gathered a large dataset on daily bike demands across the American market based on some factors.

**Business Goal**:

You are required to model the demand for shared bikes with the available independent variables. It will be used by the management to understand how exactly the demands vary with different features. They can accordingly manipulate the business strategy to meet the demand levels and meet the customer's expectations. Further, the model will be a good way for management to understand the demand dynamics of a new market.

## Table of Contents

- [General Info](#general-information)
- [Technologies Used](#technologies-used)
- [Acknowledgements](#acknowledgements)


## General Information

**Issues to Address:**

1. **Revenue Decline:** BoomBikes faces substantial revenue declines due to the ongoing pandemic, necessitating a strategic solution.
2. **Market Sustainability:** The company struggles to sustain itself in the current market scenario, demanding a mindful business plan.
3. **Post-Lockdown Strategy:** BoomBikes aims to accelerate revenue post-lockdown, requiring an understanding of post-quarantine bike demand.

**Objectives:**
The objectives include predicting significant variables influencing American market shared bike demand, determining the crucial predictors, developing a model to understand demand variations, facilitating adaptive business strategies, and exploring demand dynamics for effective decision-making. This case study aims to achieve this goal by building a multivariate linear regression model using the provided [dataset](./day.csv).

The notebook appears to cover a comprehensive bike-sharing assignment, including data handling, analysis, and modeling. Here's a point-wise summary based on the content:

**Importing Libraries:**

Standard data manipulation libraries (numpy, pandas).
Visualization tools (matplotlib.pyplot, seaborn).
Machine learning libraries (sklearn for model building and evaluation, statsmodels for statistical analysis).
Warning suppression using warnings.filterwarnings('ignore').
Dataset Loading and Initial Exploration:

**Read Dataset:**

The dataset day.csv is read into a DataFrame.
Basic exploration with .head(), .describe(), .shape to understand structure, summary statistics, and dimensions.
Check for data types, null values, and unique values.

**Data Cleaning and Preprocessing:**

Likely steps include handling missing values, renaming columns for clarity, and checking data consistency.
Feature engineering or transformations might be applied, such as scaling (MinMaxScaler, StandardScaler).

**Exploratory Data Analysis (EDA):**

Visualization of trends, distributions, and correlations using seaborn and matplotlib.
Identification of key patterns, seasonal trends, and relationships between variables.

**Feature Selection:**

Techniques like Recursive Feature Elimination (RFE) and Variance Inflation Factor (VIF) are employed to select important features and handle multicollinearity.

**Model Building:**

Linear Regression and ElasticNet models are built to predict bike-sharing counts.
Data splitting into training and test sets using train_test_split.

**Model Evaluation:**

Performance is evaluated using metrics like R², Mean Absolute Error (MAE), and Mean Squared Error (MSE).
Cross-validation and GridSearchCV are used to optimize model parameters.

**Statistical Analysis:**

Regression analysis using statsmodels for detailed statistical insights.
Examination of coefficients, p-values, and confidence intervals.

**Conclusion:**

Final insights or recommendations based on model results and data analysis.
- Three key feature variables, **temp**, **yr**,, and **windspeed**, exhibit the highest coefficient values, indicating their significant impact.

## Technologies Used

- [Python](https://www.python.org/)
- [Matplotlib](https://matplotlib.org/)
- [Numpy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/)
- [Seaborn](https://seaborn.pydata.org/)
- [Statsmodels](https://www.statsmodels.org/stable/index.html)
- [Scikit-Learn](https://scikit-learn.org/stable/)

## Conclusions

- The equation of the best fit line is given by:

- The close alignment of R2 and adjusted R2 values between the training and test sets (R2: 0.9141 vs. 0.8761 and Adjusted R2: 0.9093 vs. 0.8691) in a linear regression model indicates effective generalisation. This similarity suggests the model avoids overfitting to the training data and is likely to perform consistently on new, unseen data.

- Bike demand is influenced by features such as **year**, **temp**, **wathersit**, **mnth**, **windspeed**

- Three key feature variables, **temp**, **yr**, and **wathersit**, exhibit the highest coefficient values, indicating their significant impact.

## Acknowledgements

- This project was inspired by live session of upGrad Relevance of Linear Regression Models

## Contact

Created by [@NavdeepMehta](https://github.com/NavdeepMehta)
