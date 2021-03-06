# Data-Science-Portfolio

## [1. Boat Price Prediction](https://github.com/teolj96/Boats_Project_repo)

## About
In this project I scraped the web for information about various boats available for sale online. As a potential customer looking to buy a sailing boat one day my goal was to see if I can mine useful information from the data and help me in my decision making. Upon this data I also built a machine learning model using linear regression and ANNs which can predict the boat price. A large portion of the work went on scraping the correct information and cleaning the data afterwards.

Click the title for more information.

## Methods used
* Web scraping using Scrapy
* Data cleaning
* EDA
* Feature enginnering
* Model building

## Key findings
* Motor yachts are the most popular type of boat, followed by sailing boats and sport boats
* Germany is the most popular location for boat brokers, followed by Switzerland and Italy. For the fact that Switzerlands doesn't have access to the sea, this is very impressive
* Sunseeker manufactures the most expensive yachts
* Price is highly correlated with Length and Width, which makes sense because the boat size is the strongest determining factor of the price

## Some key graphs
![alt text](pictures/boat_model.jpg "Graph 1")

Picture 1: Most popular boat brands

![alt text](pictures/boat_type.jpg "Graph 2")

Picture 2: Most popular types of boat

![alt text](pictures/price_material.jpg "Graph 3")

Picture 3: Boat price by material

![alt text](pictures/lin_reg.jpg "Graph 4")

Picture 4: Linear regression model

## Model performance

| Model         | MAE         | MSE         |
| ------------- |-------------|-------------|
| Linear Regression      | 45667.50| 5200817068.73 |
| ANN (Input: 121, Hidden: 121, Output: 1) | 34381.004      |9830265000.0|
| ANN (Input: 121, Hidden: 121, Hidden: 121, Output: 1)   | 34995.54 | 14412964000.0 |
| ANN (Input: 121, Hidden: 242, Output: 1)           | **34628.77**      | **10111264000.0** |
| ANN (Basic model with SGD)                     | 45051.16 | 8585146000.0 |

As expected ANN turned out to be a more powerful and effective machine learning model for predicting the price of boats, but even with hyperparameter tuning the model couldn't go better then a MAE score of 34000, meaning that doing more feature engineering and reducing the number of features would be the next step in model optimization.

## [2. Spaceship Titanic Classification Prediction](https://github.com/teolj96/Data-Science-Portfolio/blob/main/Spaceship%20Titanic%20(GridSearch%2C%20Pipeline%2C%20LogReg%2C%20XGB%2C%20SVM%2C%20NN%2C%20RandomForest%2C).ipynb)

## About
The Spacehip Titanic dataset is a [Kaggle](https://www.kaggle.com/c/spaceship-titanic/data) competition dataset based on the famous original Titanic dataset, this time set in future. The goal of the project was explore which features influence the most a persons ability to get transported and create the best possible model which could predict that. 

## Methods used
* Exploratory data analysis
* Data cleaning
* Feature engineering
* Using pipelines
* Model building and optimization

## Key findings
Through data exploration and data visualization I found out that:

  * Being from planet Europa

  * Being in Cryosleep

  * Traveling to Trappist or Cancri

  * Being of younger age

  * Being richer/spending more

Gives a person better chance of being transported on the spaceship.

## Some key graphs
![alt text](pictures/01.JPG "Graphs 1")

Graph 1: There is a higher change to be transported if you are from planet Europa

![alt text](pictures/02.jpg "Graphs 2")

Graph 2: Travelling to TRAPPIST or Cancri gives you higher change to be transported

![alt text](pictures/03.jpg "Graphs 3")

Graph 3: The age distribution of people who were or were not transported

![alt text](pictures/04.jpg "Graphs 4")

Graph 4: Correlation heatmap

## Model performance

| Model         | Accuracy          
| ------------- |-------------|
| Logistic Regression      | 0.7779|
| Random Forest            | **0.8033**      |
| SVM                      | 0.7612
| XGB Classifier           | **0.8039**      |
| ANN                      | 0.8016

I was able to get very similiar score of around 80%, the best being XGB Classifier by a small margain. I was not able to go much more than 80% with optimization. The only thing that might have helped was reducing the number of total features in the final dataset.


## [3. Wine Quality Classification Prediction](https://github.com/teolj96/Data-Science-Portfolio/blob/main/Wine%20Classification%20(StandardScaler%2C%20ImbalancedLearn%2C%20SMOTE%2C%20XGB).ipynb)

## About
In this project I used a [Kaggle](https://www.kaggle.com/yasserh/wine-quality-dataset) dataset to create a classification model that could predict the quality of the wine by their unique features and qualities.

## Methods used
* Exploratory data analysis
* Data cleaning
* Model building
* SMOTE oversampling
* Model optimization

## Some key graphs
![alt text](pictures/wine_02.jpg "Graph 1")

Graph 1: The overall distribution of the features was generally equal

![alt text](pictures/wine_03.jpg "Graph 2")

Graph 2: But there were still a lot of outliers that needed to be cleaned

![alt text](pictures/wine_04.jpg "Graph 3")

Graph 3: Correlation heatmap

![alt text](pictures/wine_01.jpg "Graph 4")

Graph 4: Piechart showing the inbalance in the labels

## Model performance

| Model         | Accuracy          
| ------------- |-------------|
| Decision Tree      | 0.7701|
| Random Forest            | **0.8551**      |
| KNN                      | 0.8367
| XGB Classifier           | 0.8459      |

In the end the Random Forest Classifier proved to be the most successful with an accuracy score of 0,86. The next best model was the XGB Classifier with an accuracy score of 0,84. The KNN model with only 1 neighbour showed a good accuracy score but such model would have a high bias and would not be very realistic, thus it isn't of use to us.


## [4. XGB Regressor Time Series Project](https://github.com/teolj96/Data-Science-Portfolio/blob/main/Store%20sales%20time%20series%20(Data%20Cleaning%2C%20Groupby%2C%20EDA%2C%20XGBoost).ipynb)

## About
In this project I analysed the Store Sales [Kaggle](https://www.kaggle.com/c/store-sales-time-series-forecasting) competition dataset. It consists of several datasets such as: train, test, oil, holidays, stores and oil prices. The goal of the project was to analyse and extract various information and create a Machine Learning model that could predict store sales. 

## Methods used
* Exploratory data analysis
* Data cleaning
* Feature engineering (encoding)
* XGB Regression

## Key findings
* Peak sales happen in December
* Most sales are generated by stores numbers 40-50
* Buyers preffer the weekends over the weekdays
* City of Ouito has the highest sales numbers

## Key graphs
![alt text](pictures/sales_02.jpg "Graph 1")

Graph 1: Sales per month

![alt text](pictures/sales_03.jpg "Graph 2")

Graph 2: Sales by store family

![alt text](pictures/sales_04.jpg "Graph 3")

Graph 3: Sales by state

![alt text](pictures/sales_05.jpg "Graph 4")

Graph 4: Linear model visualized

## Model performance

| XGB         | Results          
| ------------- |-------------|
| Accuracy      | 0.9364   |
| MAE            | 95.1703      |
| MSE                      | 76950.7888
| RMSE           | 277.4001      |
