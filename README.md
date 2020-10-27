# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
This dataset contains data about direct marketing campaigns of a banking institution. The marketing campaigns were based on phone calls. So we seek to predict if the product (bank term deposit) would be ('yes') or not ('no') subscribed.

The problem was solved trying two powerful methods. By one hand we train several models using the Hyperdrive tool using RandomParameterSampling to thes different parameter combinatios. On the other hand we use the AutoML SDK for training several kinds of models such as LightGBM, XGBoost, Logistic Regression, VotingEnsemble, among others. For both cases, we built a pipeline.

## Scikit-learn Pipeline
**Explain the pipeline architecture, including data, hyperparameter tuning, and classification algorithm.**
For this project we built two pipeline, each for tool as we see ine the following image:
![architecture](/image/creating-and-optimizing-an-ml-pipeline.jpg)

**What are the benefits of the parameter sampler you chose?**

**What are the benefits of the early stopping policy you chose?**

## AutoML
**In 1-2 sentences, describe the model and hyperparameters generated by AutoML.**

## Pipeline comparison
**Compare the two models and their performance. What are the differences in accuracy? In architecture? If there was a difference, why do you think there was one?**
By one hand we train several models using the Hyperdrive tool, in wich the best model resulted a Logistic Regression with the parameters C = 1.14 and max_iter = 100. The Accuracy of this model was 0.9153. On the other hand we use the AutoML SDK for training several kinds of models such as LightGBM, XGBoost, Logistic Regression, VotingEnsemble, among others. The best model of automl turned out a Voting Ensemble with AUC_weighted = 0.952 and Accuracy = 0.9146.

## Future work
**What are some areas of improvement for future experiments? Why might these improvements help the model?**

## Proof of cluster clean up
**If you did not delete your compute cluster in the code, please complete this section. Otherwise, delete this section.**
**Image of cluster marked for deletion**

## Sources:
https://archive.ics.uci.edu/ml/datasets/Bank+Marketing
