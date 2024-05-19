# DSC232R-Spring-2024---Steam-Reviews
Analysis of Steam user reviews

By Chase Farrell, Lauren Marrs, Alison Cher, Sahra Ranjbar

Abstract:
Our team is seeking to explore the “100 Million+ Steam Reviews” dataset from Kaggle. This dataset contains 24 columns and over 100 million+ reviews.Steam allows users to review their gamers, and they can either leave a positive or negative review; other users can also rate their reviews. The data contains information on the users submitting the reviews (play time, games owned, reviewing history), as well as statistics pertaining to the review itself (upvotes, comments, positive/negative). Using this dataset we’d like to explore the relationship between steam players and their reviews.Our team is mainly interested in how playtime (all time, over last 2 weeks, at time of review, or when they last played) influences a users rating and how helpful other users find their review. Since our dataset is ~40 GBs, we will be using Pyspark to answer our questions. 

Kaggle link to our data:
https://www.kaggle.com/datasets/kieranpoc/steam-reviews/data 

When loading a SDSC Jupyter session we used:

4 cores

8 GB per node

The jupyter notebook contains code blocks to download the dataset.

Initial preprocessing summary:
1. filtered reviews to english
2. filtered voted_up to 0 and 1
3. selected specific columns of interest

Next we took a random .5% sample of the >100 million reviews, resulting in a 200,000 row sample to work with.
We put this sub-sample into a pandas dataframe to visualize.
From this sample, we converted the values to integers, as the whole table was stored as strings.

The jupyter notebook contains several visualizations, including histograms and scatterplots comparing play time to up/down votes.

NOTE: the x-axis for play time is on a logarithmic scale

Milestone 3 

The first implementatioin of our model uses logistic model. This was a logical choice as we are attempting to predict a binary variable of whether a user "upvotes" a particular game with their review based on various steam user metrics ( ---- ). The ground truth of our model that we are assuming, is that as user play the game more (at review, overall, or over the last two weeks), we should expect that steam users would be more postive. Vice versa, if they play the game less we'd expect that we'd see the opposite. Our model achieves ~86% accurancy in our training set, and ~68% in our test set, this is likely a sign that our model is overfitting to our training set.

Our current model is farily uncomplex as it is a logistic model, but there seems to be a fairly large difference between the error in the training data and the error in the test data. Given the uncomplex nature of our model, I'd expect it to fit more on the left of the fitting graph; however, given the results and the inputs we are using from the dataset. Perhaps there is autocorrelation between our variables that is causing some amount of overfitting, afterall, the playtime in the last two weeks and the overall playtime may be correlated with each other. Perhaps we should look to include other features in our model and more selectively choose which temporal features we are using to predict the upvotes of our steam reviews. Although our model seems to confirm our ground truths, that more hours seems to predict more positive reviews, we'd like to get a more robust result, improvements to the model could be simply what features we should select, or limit features that may be autocorrelated.

Given these results for further models we may want to consider, like decision trees and random forests. Our current model assumses a linear relationship between our features and the log-odds outcome, we may want to consider a decision tree as it is more interpretable than our current choice. Although a decision tree could also be prone to overfitting, may want to start simple with our logistic model and continue to add complexity with decision trees and then random forests. Random forests are robust to overfitting which seems to be a difficulty in our current model.

