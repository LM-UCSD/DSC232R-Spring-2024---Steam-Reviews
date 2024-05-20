# DSC232R-Spring-2024---Steam-Reviews
Analysis of Steam user reviews

By Chase Farrell, Lauren Marrs, Alison Cher, Sahra Ranjbar


When loading a SDSC Jupyter session we used:

4 cores

8 GB per node

The data set is from https://www.kaggle.com/datasets/kieranpoc/steam-reviews/data

Instructions on how to download the data and set up the environment can be found in Milestone-2
https://github.com/LM-UCSD/DSC232R-Spring-2024---Steam-Reviews/tree/Milestone-2

Initial preprocessing summary:
1. filtered reviews to english
2. filtered voted_up to 0 and 1
3. selected specific columns of interest
4. corrected data types from string to numeric as needed


Milestone 3 

The first implementatioin of our model uses logistic model. This was a logical choice as we are attempting to predict a binary variable of whether a user "upvotes" a particular game with their review based on various steam user metrics (Playtime in the last two weeks, at review, and overall). The ground truth of our model that we are assuming, is that as user play the game more (at review, overall, or over the last two weeks), we should expect that steam users would be more postive. Vice versa, if they play the game less we'd expect that we'd see the opposite. Our model achieves ~86% accurancy in our training set, and ~64% in our test set, this is likely a sign that our model is overfitting to our training set.

Our current model is farily uncomplex as it is a logistic model, but there seems to be a fairly large difference between the error in the training data and the error in the test data. Given the uncomplex nature of our model, we would expect it to fit more on the left of the fitting graph; however, given the results and the inputs we are using from the dataset, perhaps there is autocorrelation between our variables that is causing some amount of overfitting, such as the possibility that the playtime in the last two weeks and the overall playtime may be correlated with each other. Perhaps we should look to include other features in our model and more selectively choose which temporal features we are using to predict the upvotes of our steam reviews. 

Conclusions:
Although our model seems to confirm our ground truths, that more hours seems to predict more positive reviews, we'd like to get a more robust result, improvements to the model could be simply what features we should select, or limit features that may be autocorrelated.

Given these results for further models we may want to consider, like decision trees and random forests. Our current model assumses a linear relationship between our features and the log-odds outcome, we may want to consider a decision tree as it is more interpretable than our current choice. Although a decision tree could also be prone to overfitting, may want to start simple with our logistic model and continue to add complexity with decision trees and then random forests. Random forests are robust to overfitting which seems to be a difficulty in our current model.

