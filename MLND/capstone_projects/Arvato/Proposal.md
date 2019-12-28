# Customer Segmentation Report for Arvato Financial Services

## domain background
	- This project is the capstone project for machine learning nano-degree. We need to use unsupervised and supervised techniques to analyze the demographics data. The goal of this project is to characterize the customers segment of population and to build a model to make predictions.
	- The data for this project is provided by Bertelsmann Arvato Analytics, and represents a real-life data science project. 
	

## problem statement

	 1. Customer Segmentation Report
		- Access the first and second dataset by using unsupervised learning methods to analyze attributes of established customers and the general population in order to create customer segments.

	 2. Supervised Learning Model
	 	- Use the choosed features from the part 1 to build a machine learning model that predicts whether or not each individual will respond to the campaign, based on the third dataset.

	 3. Kaggle Competition
		- Use the model built in the second part to make predictions on the campaign data as (the fourth dataset) part of a Kaggle Competition. 

## datasets and inputs

There are four data files associated with this project:

	1. Udacity_AZDIAS_052018.csv: Demographics data for the general population of Germany; 891 211 persons (rows) x 366 features (columns).
	2. Udacity_CUSTOMERS_052018.csv: Demographics data for customers of a mail-order company; 191 652 persons (rows) x 369 features (columns).
	3. Udacity_MAILOUT_052018_TRAIN.csv: Demographics data for individuals who were targets of a marketing campaign; 42 982 persons (rows) x 367 (columns).
	4. Udacity_MAILOUT_052018_TEST.csv: Demographics data for individuals who were targets of a marketing campaign; 42 833 persons (rows) x 366 (columns).

In addition to the above data, there are two additional meta-data:

	5. DIAS Information Levels — Attributes 2017.xlsx: a top-level list of attributes and descriptions, organized by informational category
	6. DIAS Attributes — Values 2017.xlsx: a detailed mapping of data values for each feature in alphabetical order

## solution statement

	1. Data preprocessing is the first and essential step, mainly to deal with the issing/unknown value in the dataset and re-encode the features. I will abide by the workflow of data analysis learned from DAND to preprocessing those datasets.

	2. Customer Segmentation Report 
	In this part, first of all, I will use the most common dimensions reduction method is PCA (Principal Component Analysis) to try to reduce the dimensions of the dataset. And second of all, I will use the K-meanss clustering method to perform unsupervised learning to seperate the general population into a few groups. 

	3. Supervised Learning Model 
	I am going to use essemble methods. Use a few different Regressor like Light Gradient Boosting Regressor, XGBoost Regressor, Support Vector Regressor, etc. And then stack them up and optimize them, remove the weakest model. Blend them together to get the best performance.

## benchmark model

In this project [kaggle](https://www.kaggle.com/c/udacity-arvato-identify-customers/leaderboard) competition public leaderboard, the best score(AUC) is 0.80819. This is going to be the bechmark model.


## evaluation metrics

In the segmentation part, explained variance ratio is be used in the PCA process. 

![Explained Variance](https://render.githubusercontent.com/render/math?math=%5Cbegin%7Bequation%2A%7D%0A%5Cfrac%7B%5Csum_%7Bn%7D%5E%7B%20%7D%20s_n%5E2%7D%7B%5Csum%20s%5E2%7D%0A%5Cend%7Bequation%2A%7D&mode=display) 
[image source](https://github.com/udacity/ML_SageMaker_Studies/blob/master/Population_Segmentation/Pop_Segmentation_Exercise.ipynb)

Explained variance accounts for the ability to describe the whole feature variance, the more the explained variance, the more import of the component.
In the supervised model prediction parts, mean squared log error and AUC are used as main metric.

![MSLE math](https://peltarion.com/static/msle_01.png) 
[image source](https://peltarion.com/knowledge-center/documentation/modeling-view/build-an-ai-model/loss-functions/mean-squared-logarithmic-error)

![Area under the curve](https://miro.medium.com/max/361/1*pk05QGzoWhCgRiiFbz-oKQ.png) 
[image source](https://towardsdatascience.com/understanding-auc-roc-curve-68b2303cc9c5)

## project design
	
	1. Exploratory Data Analysis (EDA):
		- To get some initial insight from the data. Data cleaning, data wrangling, data visulization will be done. 

	2. Dimensionality reduction with PCA:
		- It is very hard for K-means to figure out which features are most important in a higher dimensions. So before clustering this data, PCA will be hired to reduce the number of features within a dataset. Try to retain the "pricipal components".

	3. Clustering data with k-means:
		- Use the unsupervised clustering algorithm, K-means to segment customers using their PCA attributes. Will use 'Elbow Graph' to guide me to find a "good" K.

	4. Supervised modeling:
		- Define and train a binary logistic classfier to effectively separate two classes of MAILOUT data.
