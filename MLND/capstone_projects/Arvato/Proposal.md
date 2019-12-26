# Customer Segmentation Report for Arvato Financial Services

## domain background

	

## problem statement

 1. Customer Segmentation Report
This section will be similar to the corresponding project in Term 1 of the program, but the datasets now include more features that you can potentially use. You'll begin the project by using unsupervised learning methods to analyze attributes of established customers and the general population in order to create customer segments.

 2. Supervised Learning Model
You'll have access to a third dataset with attributes from targets of a mail order campaign. You'll use the previous analysis to build a machine learning model that predicts whether or not each individual will respond to the campaign.

 3. Kaggle Competition
Once you've chosen a model, you'll use it to make predictions on the campaign data as part of a Kaggle Competition. You'll rank the individuals by how likely they are to convert to being a customer, and see how your modeling skills measure up against your fellow students.

## datasets and inputs

There are four data files associated with this project:

	- Udacity_AZDIAS_052018.csv: Demographics data for the general population of Germany; 891 211 persons (rows) x 366 features (columns).
	- Udacity_CUSTOMERS_052018.csv: Demographics data for customers of a mail-order company; 191 652 persons (rows) x 369 features (columns).
	- Udacity_MAILOUT_052018_TRAIN.csv: Demographics data for individuals who were targets of a marketing campaign; 42 982 persons (rows) x 367 (columns).
	- Udacity_MAILOUT_052018_TEST.csv: Demographics data for individuals who were targets of a marketing campaign; 42 833 persons (rows) x 366 (columns).

In addition to the above data, there are two additional meta-data:
DIAS Information Levels — Attributes 2017.xlsx: a top-level list of attributes and descriptions, organized by informational category
DIAS Attributes — Values 2017.xlsx: a detailed mapping of data values for each feature in alphabetical order

## solution statement

Data preprocessing
	Deal with Missing/Unknown Value
	Re-Encode and pick up features
Customer Segmentation Report 
	PCA
	K-mean clustering
Supervised Learning Model 
	Light Gradient Boosting Regressor
	XGBoost Regressor
	Ridge Regressor
	Support Vector Regressor
	Random Forest Regressor
	gradientboosting

## benchmark model



## evaluation metrics

In the segmentation part, explained variance ratio is be used in the PCA process. Explained variance accounts for the ability to describe the whole feature variance, the more the explained variance, the more import of the component.
In the supervised model prediction parts, accuracy is used as main metric.

## project design

