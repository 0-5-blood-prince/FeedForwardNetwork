# FeedForward Neural Network
Classification Task on Fashion MNIST Dataset using a FeedForward Neural Network

Libraries used:
1) Wandb
2) Numpy
3) scipy
4) scikit-learn
5) seaborn
6) matplotlib

Dataset.py:
This file loads the dataset and does normalisation.
Logging Images for Q1 is also implemented here.

NeuralNet.py:
This contains the entire neural net framework, which includes Forward and backward propogation, all required optimizers.

When you initialize the network with the required parameters and call the fit() function, it trains the network model using the given data.

Experimenter.py:
This uses wandb for conducting Hyper Parameter Search for finding the best model using the sweep config.

Plots.py:
This contains code for generating various plots asked in the assignment.

Steps to follow for Evaluation/Reciprocate the resutls in report:
Question 1,7,8,10:
    Run: "python plots.py QX"
    to get results in report for Question 'X'
Question 4,5,6:
    Run: "python experimenter.py"
