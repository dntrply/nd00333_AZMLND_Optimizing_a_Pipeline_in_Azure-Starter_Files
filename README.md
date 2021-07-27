# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
**This dataset comprises data about a marketing campaign at a bank. The goal is to determine
if a customer will subscribe to a Term Deposit offer**

**The best performing model is the scikit-learn LogisticRegression model with an accuracy of 0.9162**

## Scikit-learn Pipeline
The pipeline architecture uses AzureML HyperDrive. HyperDrive allows the user to specify the training script and script environment, the input data to the training model, the metric to improve (**accuracy**) and the hyperparameters for the model (**--C** - inverse of regularization strength and **--max_iter** - maximum number of iterations). Parameter Sampling algorithms (eg. **random parameter sampling**) and early stopping  (eg. **bandit policy**) algorithms may also be specified. HyperDrive then repeatedly executes the training script with a combination of hyperdrive parameters with the intent to maximize the best metric. Metrics can readily be retrieved from the best run and the model readily persisted for later deployment or retraining. The training algorithm chosen is the **scikit-learn LogisticRegression** classification algorithm. It is suited for this dataset where we classify whether the customer will or will not subscribe to a term deposit.

**Benefits of RandomParameterSampling (chosen parameter sampler)**
RandomParameterSampling allows for a range of combinations of parameter values to be tested. The benefit is that the user is not limited to guessing and tracking a handful of combination of values. Rather, the system attempts multiple combinations and keeps track of the parameters and the results.

**Benefits of BanditPolicy (chosen early stopping policy)**
The early stopping policy chosen was the Bandit Policy. The Bandit Policy allows you to specify a slack factor. The run ends when the primary metric for a run
is not within the specified slack factor of the most successful run. This allows the training to stop if metric improvement is unlikely.

## AutoML
**Best Model**
The best AutoML model is VotingEnsemble, comprising of a data transformer and an ensemble of many weighted models. The DataTransformer performs automatic featurization. The highest weighted model is the MaxAbsAcaler, LightGBM classifier model. LighGBM Model has hyperparameters such as min_data_in_leaf

## Pipeline comparison
The scikit-learn LogisticRegression model with an accuracy of **0.9162** performed better than the best model from AutoML with accuracy **0.9152**. The sklearn LogisticRegression model likely is the better model for the nature of data provided in this dataset. Although AutoML tries a number of algorithms, it was unable to out-perform the scikit-learn LogisticRegression model. 

AutoML is more hands-off and more comprehensive whereas Hyperdrive is more customizable as one may provide a custom training script.

## Future work
* Providing balanced data with a number of positive outcomes closer to the number of negative outcomes may also improve the autoML model.
* More training data may also improve the experiments.
