# winequality

# Wine Quality Prediction Project, Machine Learning Capstone Project

This repository contains code and associated files for defining and deploying classification models for red wine quality using AWS SageMaker.

## Project Overview
In this project, I'll look at a winequality dataset, and build a binary classification model that can that can identify which physiochemical properties make a wine to be classified as either a ‘good wine’ or ‘bad wine’.It is based on the study of P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.

**Labeled Data**
The red wine quality dataset (P. Cortez et al. 2009) was downloaded from Kaggle. This has 11 features (11 physiochemical properties) and labels (quality rankings) of different red wines from Portugal. The labels are from 1 (very bad) to 10 (very excellent). There are 1599 samples. According to repository page on Kaggle, the classes are ordered and not balanced (e.g. there are much more normal wines than excellent or poor ones). The features are:

fixed acidity
volatile acidity
citric acid
residual sugar
chlorides
free sulfur dioxide
total sulfur dioxide
density
pH
sulphates
alcohol
In this notebook, I'd like to train a model based on these features so that we can predict the quality of a red wine in the future.

**Binary Classification**
Since we have true labels to aim for, we'll take a supervised learning approach and train a binary classifier to sort data into one of our two transaction classes: fraudulent or valid. We'll train a model on training data and see how well it generalizes on some test data.

The notebook will be broken down into a few steps:

* Loading and exploring the data
* Splitting the data into train/test sets
* Defining and training a LinearLearner, binary classifier
* Making improvements on the model
* Defining a custom sklearn SVM model and training script
* Evaluating and comparing models test performance
* Retrieving the five most informative features

A lot of this notebook will focus on making improvements. Specifically, I'll address techniques for:

1. Tuning a model's hyperparameters and aiming for a specific metric, such as high recall or precision.
2. Managing class imbalance, since we have in this case, much more normal wines than excellent or poor ones. In the later part of this notebook, I will try to compare the performance of LinearLearner and a custom sklearn SVM model.


---
## Installation
Please see the [README](https://github.com/udacity/ML_SageMaker_Studies/tree/master/README.md) in the root directory for instructions on setting up a SageMaker notebook


## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)


