# DSC232R-Spring-2024---Steam-Reviews
Analysis of Steam user reviews

By Chase Farrell, Lauren Marrs, Alison Cher, Sahra Ranjbar

Milestone 1:

Abstract:
Our team is seeking to explore the “100 Million+ Steam Reviews” dataset from Kaggle. This dataset contains 24 columns and over 100 million+ reviews.Steam allows users to review their gamers, and they can either leave a positive or negative review; other users can also rate their reviews. The data contains information on the users submitting the reviews (play time, games owned, reviewing history), as well as statistics pertaining to the review itself (upvotes, comments, positive/negative). Using this dataset we’d like to explore the relationship between steam players and their reviews.Our team is mainly interested in how playtime (all time, over last 2 weeks, at time of review, or when they last played) influences a users rating and how helpful other users find their review. Since our dataset is ~40 GBs, we will be using Pyspark to answer our questions. 

Kaggle link to our data:
https://www.kaggle.com/datasets/kieranpoc/steam-reviews/data 

Milestone 2:

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

The first implementatioin of our model uses logistic model. This was a logical choice as we are attempting to predict a binary variable of whether a user "upvotes" a particular game with their review based on various steam user metrics (Playtime in the last two weeks, at review, and overall). The ground truth of our model that we are assuming, is that as user play the game more (at review, overall, or over the last two weeks), we should expect that steam users would be more postive. Vice versa, if they play the game less we'd expect that we'd see the opposite. Our model achieves ~86% accurancy in our training set, and ~68% in our test set, this is likely a sign that our model is overfitting to our training set.

Our current model is farily uncomplex as it is a logistic model, but there seems to be a fairly large difference between the error in the training data and the error in the test data. Given the uncomplex nature of our model, I'd expect it to fit more on the left of the fitting graph; however, given the results and the inputs we are using from the dataset. Perhaps there is autocorrelation between our variables that is causing some amount of overfitting, afterall, the playtime in the last two weeks and the overall playtime may be correlated with each other. Perhaps we should look to include other features in our model and more selectively choose which temporal features we are using to predict the upvotes of our steam reviews. Although our model seems to confirm our ground truths, that more hours seems to predict more positive reviews, we'd like to get a more robust result, improvements to the model could be simply what features we should select, or limit features that may be autocorrelated.

Given these results for further models we may want to consider, like decision trees and random forests. Our current model assumses a linear relationship between our features and the log-odds outcome, we may want to consider a decision tree as it is more interpretable than our current choice. Although a decision tree could also be prone to overfitting, may want to start simple with our logistic model and continue to add complexity with decision trees and then random forests. Random forests are robust to overfitting which seems to be a difficulty in our current model.

Milestone 4

Introduction

Steam, a video game digital distribution service, hosts over 34,000 games and is one of the largest digital storefronts for video games. As part of their storefront, Steam hosts user reviews of the games they distribute. Users may leave a positive or negative review with text and other users may give feedback to that review whether it was helpful or otherwise. The portion of positive to negative reviews creates an overall metric and a recent metric ranging from Overwhelming Positive (95% over more positive), Very Positive (80% of more), Mostly Positive (70%-79%), Mixed (40%-69%), Mostly Negative (20%-39%), Overwhelmingly Negative (0%-19%). Generally, developers would prefer to have their games on the positive side as this should understandably increase sales overtime. This project attempts to provide insight into steam reviews to give developers a better understanding of why users leave negative or positive reviews. 

Using a 100+ Million Steam Review dataset sourced from Kaggle, our team seeks to better understand the relationship between steam players and their reviews. Using the data collected by the kaggle dataset, our team specifically explores the relationship between user playtime and whether they’d leave a negative or positive review. By using a logistic regression model, we attempt to predict whether a user leaves a positive or a negative review (Are we going to add a decision tree as well). The results of which allow developers to make decisions on the prioritization of user playtime in their games.

Methods

	Initial Data Analysis and Findings/Rationale for Study
	
 	Visualizations of Data

Data Exploration

To explore the data, we first checked the data type of all the features. From here, we selected features of interest and reviewed a few lines of the data table to understand the scope of the contents. After this, we pulled a .describe() command to see summary statistics of the data as well as a count of rows.
![Untitled](https://github.com/LM-UCSD/DSC232R-Spring-2024---Steam-Reviews/assets/128201733/4153bdbc-4973-4567-a838-03330677b9d7)


Preprocessing

For our initial preprocessing of the data, we first enforced the data types manually as all the data types were initially loaded as strings. We then limited the reviews to English. This is not entirely necessary for our analysis that focused mostly on playtime; however, we wanted an English audience to understand the individual reviews if necessary. Most of the reviews are in English (>49%), but our results should be limited to English speakers reviewing games for this reason. Next we filtered the “voted_up” column to only include 0 (negative review) and 1 (positive review) . This was because there were anomalies in the data; for example, we found a review in this column in one of the cells. We then limited our analysis to the voted up, playtime in the last two weeks, playtime at review, and playtime overall as we are only studying this relationship and did not need any other columns as they were not relevant for our specific study. Next we then ensured that no null values were present in our data. Finally we created a VectorAssembler set to skip invalid cells for use in PySpark machine learning.

	Steps Taken to Clean and prepare the data

	Feature selection

Model 1: Logistic Regression

The first implementatioin of our model uses logistic model. This was a logical choice as we are attempting to predict a binary variable of whether a user "upvotes" a particular game with their review based on various steam user metrics (Playtime in the last two weeks, at review, and overall). The ground truth of our model that we are assuming, is that as user play the game more (at review, overall, or over the last two weeks), we should expect that steam users would be more postive. Vice versa, if they play the game less we'd expect that we'd see the opposite.

	Parameter Choices and rationale

	Model training and validation

Model 2: Decision tree

	Parameter Choices and rationale

	Model training and validation

Results

	Model 1: Logistic Regression Results
 
Our model achieves ~86% accurancy in our training set, and ~68% in our test set, this is likely a sign that our model is overfitting to our training set.
 
		Performance/Accuracy
	
		Visualization of Model

	Model 2: Decision Tree Results
	
		Performance/Accuracy
		
		Visualization of Model

Discussion Section

	Model 1: Logistic Regression Results

 Our current model is farily uncomplex as it is a logistic model, but there seems to be a fairly large difference between the error in the training data and the error in the test data. Given the uncomplex nature of our model, I'd expect it to fit more on the left of the fitting graph; however, given the results and the inputs we are using from the dataset. Perhaps there is autocorrelation between our variables that is causing some amount of overfitting, afterall, the playtime in the last two weeks and the overall playtime may be correlated with each other. Perhaps we should look to include other features in our model and more selectively choose which temporal features we are using to predict the upvotes of our steam reviews. Although our model seems to confirm our ground truths, that more hours seems to predict more positive reviews, we'd like to get a more robust result, improvements to the model could be simply what features we should select, or limit features that may be autocorrelated.

 Given these results for further models we may want to consider, like decision trees; which we use for our second model. Our current model assumses a linear relationship between our features and the log-odds outcome, we may want to consider a decision tree as it is more interpretable than our current choice. Although a decision tree could also be prone to overfitting, may want to start simple with our logistic model and continue to add complexity with decision trees and then random forests. Random forests are robust to overfitting which seems to be a difficulty in our current model.


		Interpretation of Results

		Strengths and Weaknesses of Model

		Potential Biases and Limitations of Model

	Model 2: Decision Tree

		Interpretation of Results

		Strengths and Weaknesses of Model

		Potential Biases and Limitations of Model

	Comparison

		Compare the logistic regression and decision tree
		
	Believability
		
		Credibility

		Anomalies or unexpected findings

	Critique and Reflection

		Self-critique of model/interpretation

		Areas for improvement
		
Conclusion

	Summary

	Future Work
	
	Final Thoughts


Statement of Collaboration
Lauren Marrs: I did a lot of the editing for the write-ups, worked on planning and laying out the code to explore our topic, and contributed to the final debugging to ensure our notebooks ran successfully.

Final Model and Results and Summary

	Final Model

	Results Summary

