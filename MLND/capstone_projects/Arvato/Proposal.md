# Customer Segmentation Report for Arvato Financial Services

## domain background

	

## problem statement

	 1. Customer Segmentation Report
	Access the first and second dataset by using unsupervised learning methods to analyze attributes of established customers and the general population in order to create customer segments.

	 2. Supervised Learning Model
	 Use the choosed features from the part 1 to build a machine learning model that predicts whether or not each individual will respond to the campaign, based on the third dataset.

	 3. Kaggle Competition
	Use the model built in the second part to make predictions on the campaign data as (the fourth dataset) part of a Kaggle Competition. 

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

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mfrac>
    <mrow>
      <munderover>
        <mo>&#x2211;<!-- ∑ --></mo>
        <mrow class="MJX-TeXAtom-ORD">
          <mn>5</mn>
        </mrow>
        <mrow class="MJX-TeXAtom-ORD">
   </mrow>
      </munderover>
      <msubsup>
        <mi>s</mi>
        <mi>n</mi>
        <mn>2</mn>
      </msubsup>
    </mrow>
    <mrow>
      <mo>&#x2211;<!-- ∑ --></mo>
      <msup>
        <mi>s</mi>
        <mn>2</mn>
      </msup>
    </mrow>
  </mfrac>
</math>

Explained variance accounts for the ability to describe the whole feature variance, the more the explained variance, the more import of the component.
In the supervised model prediction parts, mean squared log error and AUC are used as main metric.

![MSLE math](https://peltarion.com/static/msle_01.png)

![Area under the curve](https://en.wikipedia.org/wiki/File:ROC_curves.svg)

## project design
	- PCA 
	- K-mean clustering

	- Light Gradient Boosting Regressor
	- XGBoost Regressor
	- Ridge Regressor
	- Support Vector Regressor
	- Random Forest Regressor
	- gradientboosting