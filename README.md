# DSC232R-Spring-2024---Steam-Reviews
Analysis of Steam user reviews

By Chase Farrell, Lauren Marrs, Alison Cher, Sahra Ranjbar


When loading a SDSC Jupyter session we used:

4 cores

16 GB per node

The data set is from https://www.kaggle.com/datasets/kieranpoc/steam-reviews/data

Instructions on how to download the data and set up the environment can be found in Milestone-2
https://github.com/LM-UCSD/DSC232R-Spring-2024---Steam-Reviews/tree/Milestone-2

Initial preprocessing summary:
1. filtered reviews to english
2. filtered voted_up to 0 and 1
3. Corrected Data Types for all columns
4. Selected specific columns of interest


Milestone 3 

The first implementation of our model uses logistic regression. This was a logical choice as we are attempting to predict a binary variable of whether a user "upvotes" a particular game with their review based on various Steam user metrics (playtime in the last two weeks, at review, and overall). The ground truth assumption of our model is that as users play the game more (at review, overall, or over the last two weeks), we should expect that Steam users would be more positive in their reviews. Conversely, if they play the game less, we'd expect the opposite.

First, we used a model with a log loss metric and then an accuracy metric to evaluate the complexity of our model by examining the regularization parameter. In general, the model error rate sits ~39% for both test and train error in the log loss logistic model and stabilizes at around 13.58% for training error and 13.59% for test error in the accuracy logistic regression model.

Our current model is fairly simple as it is a logistic regression model. In the log loss case, increasing the complexity (using a smaller regularization parameter) resulted in improvements to our test error, suggesting that we are likely to the left of the fitting graph and that complexity could improve our log loss model. However, in the case of the accuracy model, there seemed to be no improvement when increasing the size of the regularization parameter, as the error rates were stable and increased slightly in our smallest regularization parameter. In both cases our errors were fairly close to one another, suggesting we may be able to benefit from increased complexity in our models; although just a regularization parameter may not be the appropriate measure of complexity.

We may want to consider the choice of our predictor variables, as many of them—such as playtime in the last two weeks and overall playtime—may be autocorrelated. One could argue that high playtime in the last two weeks is correlated with higher overall playtime; after all, as playtime in the last two weeks increases, overall playtime also increases. Therefore, we may want to consider selecting different or fewer variables that are not dependent on one another, as they could be causing additional error.

Conclusions

Although our model seems to confirm our ground truth that more hours played predicts more positive reviews, we aim to achieve more robust results. Improvements to the model could involve selecting more relevant features or limiting features that may be autocorrelated.

Given these results, we might want to consider other models such as decision trees and random forests. Our current model assumes a linear relationship between our features and the log-odds outcome. A decision tree could provide more interpretability than our current choice. Although a decision tree could also be prone to overfitting, it may be beneficial to start with our logistic model and then add complexity with decision trees and random forests. Random forests, being robust to overfitting, could address some of the difficulties we are experiencing with our current model.

