# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
This dataset contains data about direct marketing campaigns of a banking institution. The marketing campaigns were based on phone calls. So we seek to predict if the product (bank term deposit) would be ('yes') or not ('no') subscribed.

The problem was solved trying two powerful methods. By one hand we trained a Logistic Regression using the Hyperdrive tool with RandomParameterSampling to test different parameter combinatios. On the other hand we used the AutoML SDK for training several kinds of models such as LightGBM, XGBoost, Logistic Regression, VotingEnsemble, among others algorithms. For both cases, we built a pipeline.

## Scikit-learn Pipeline
For this project I built two pipeline, one for each tool as we can see in the following image:

![architecture](/image/creating-and-optimizing-an-ml-pipeline.png)

First I wrote a training script in wich the data is obtained using TabularDatasetFactory. Then I built an estimator that specifies the location of the script, sets up its fixed parameters, including the compute target and specifies the packages needed to run the script. We saw how Azure Machine Learning can help us to automate the process of hyperarameter tuning, so I launched multiple runs with different values for numbers in the sequence. I defined the parameter space using random sampling.

One of the The benefits of the random sampling is that the hyperparameter values are chosen from a set of discrete values or a distribution over a continuous range. So it tested several cases and not every combinations. It helped to reduce the time of hyperparameter tuning.

I used the BanditPolicy as early stopping policy because it defines an early termination policy based on slack criteria, frequency and delay interval for evaluation. Any run that doesn't fall within the slack factor or slack amount of the evaluation metric with respect to the best performing run will be terminated.

On this dataset we had to run classification algorithms. For the hyperdrive tuning we used a Logistic Regression and tested different combinations of parameters C and max_iter.

## AutoML
On the AutoML experiment first I defined the automl_config setting up parameters like the classification task, the primary metric as AUC_weighted, number of cross validation as 5 and the minutes of timeout in order to stop the experiment at that time. The autoML tested a several algorithms and also hyperparameters.

## Pipeline comparison
By one hand I used the Hyperdrive tool for testing different combinations of hyperparameters of C and max_iter of the Logistic Regresion algorithm.

![Hyperdrive](/image/hyperdrive_1.jpg)

![Hyperdrive](/image/hyperdrive_2.jpg)

We can see the best model is a Logistic Regression with the parameters C = 1.14 and max_iter = 100. The Accuracy of this model was 0.9153. 

On the other hand we use the AutoML SDK for training several kinds of models such as LightGBM, XGBoost, Logistic Regression, VotingEnsemble, among others algorithms. 

![Hyperdrive](/image/automl_1.jpg)

![Hyperdrive](/image/automl_2.jpg)

The best model of automl turned out a Voting Ensemble with AUC_weighted = 0.952 and Accuracy = 0.9146. However a very important advantage of the AutoML is that it also gives an explanation of the model. We can see the top importance features, in this case for example, the 'duration' variable turned out as the most important in order to predict if the product would be subscribed or not.

In general, both best models from Hyperdrive and AutoML turned out with an accuracy very similar. But may be if the Automl would have runned more time, probably it would turned out with a model with better accuracy beacuse it not only test a Logistic Regression, but also Boosting algorithms like LightGBM or Ensemble models wich are more powerfuls.

## Future work
In the case of the Hyperdrive we can test another algorithms for example LightGBM or a Neural Net in order to find the best hyperparameters. And for the AutoML we can test more time in order to find better models and leverage the Explanation tool in order to interpret the most important variables.

## Proof of cluster clean up
We can use the delete method of the AmlCompute class in order to Remove the AmlCompute object from its associated workspace.

## Sources:
https://archive.ics.uci.edu/ml/datasets/Bank+Marketing
https://classroom.udacity.com/nanodegrees/nd00333/parts/e0c51a58-32b4-4761-897b-92f6183148ac/modules/485dd1ee-e43b-468b-8e0f-60d9fe611adc/lessons/4a35baec-993d-4857-9973-85ab67ea55a2/concepts/22fbca72-0617-4c3e-84fb-5194ff28102b
https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.compute.amlcompute(class)?view=azure-ml-py#delete--
