# Data-Science-Portfolio

## [1. XGB Regressor Time Series Project](https://github.com/teolj96/Data-Science-Portfolio/blob/main/Store%20sales%20time%20series%20(Data%20Cleaning%2C%20Groupby%2C%20EDA%2C%20XGBoost).ipynb)
In this project I analysed the Store Sales [Kaggle](https://www.kaggle.com/c/store-sales-time-series-forecasting) competition dataset. It consists of several datasets such as: train, test, oil, holidays, stores and oil prices. The goal of the project was to analyse and extract various information and create a Machine Learning model that could predict store sales.

In the project I went through data cleaning with Pandas, getting rid of null values and outliers, gathering various information with Matplotlib and Seaborn about the sales, and created a XGBRegressor model with score of 0,94 and the RMSE of 277.40.

## [2. Wine Quality Classification Prediction](https://github.com/teolj96/Data-Science-Portfolio/blob/main/Wine%20Classification%20(StandardScaler%2C%20ImbalancedLearn%2C%20SMOTE%2C%20XGB).ipynb)
In this project I used a [Kaggle](https://www.kaggle.com/yasserh/wine-quality-dataset) dataset to create a classification model that could classify wine by their unique features and qualities. The dataset is relatively small and has been already cleaned before as it has no null values. I made the mistake at the beginning by cleaning outliers but it proved to be a mistake later because the dataset was so imbalanced as is presented with the pie chart in the file, but I kept the original code inside the file for future reference. The problem with cleaning the outliers is that some of the outlier were located within the smaller categories. Reducing the smaller categories even more made oversampling impossible.

I used several models such as Decision Tree, Random Forest, KNN and XGB Classifier. In the end the Random Forest Classifier proved to be the most successful with an accuracy score of 0,86. The next best model was the XGB Classifier with an accuracy score of 0,84. The KNN model with only 1 neighbour showed a good accuracy score but such model would have a high bias and would not be very realistic, thus it isn't of use to us.
