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

Data Exploration and Methods

To explore the data, we first checked the data type of all the features. From here, we selected features of interest and reviewed a few lines of the data table to understand the scope of the contents. After this, we pulled a .describe() command to see summary statistics of the data as well as a count of rows.

Next we generated figures of our desired columns of study. We went through each temporal measurement of playtime (Playtime Forever, Playtime Last Two Weeks, and Playtime at review), with whether it was upvoted or not also shown. First the play time forever
![Figure 1](https://github.com/LM-UCSD/DSC232R-Spring-2024---Steam-Reviews/assets/128201733/4153bdbc-4973-4567-a838-03330677b9d7)

This figure shows a relatively normal shape, with the majority of reviews being positive. The negative reviews seem to follow an approximately normal shape as well. In General there seems to be more reviews with more playtime, but low playtime games tend to be more negative. Other things this may show us is that there tends to be more positive reviews in our dataset. Next we show this for the Last Two weeks

![Untitled](https://github.com/LM-UCSD/DSC232R-Spring-2024---Steam-Reviews/assets/128201733/087e0adb-ee40-4dfd-9d5a-8bb12a7c9ecc)
8)

Overall, this shpae is different than the Playtime forever, there seems to be more negative reviews as playtime increase, but there is a peak at the low end of the graph. Perhaps recent playtime is not attributal to more reviews.

![Untitled](https://github.com/LM-UCSD/DSC232R-Spring-2024---Steam-Reviews/assets/128201733/26560247-06ee-4271-8eff-6f128d4f5ebb)

Next we explore playtime at review and is sseems to be a similar distribution as the playtime forever graph, this may suggest these two are related to one another. 

To futher our exploration, we do want to look for relationships with upvotes and downvotes and see if the playtimes are related to one another. Below we pair Playtime Forever and Playtime in the last two weeks together.

![Untitled](https://github.com/LM-UCSD/DSC232R-Spring-2024---Steam-Reviews/assets/128201733/44844c6f-58c0-40a0-94b1-662f1c222483)

Looking at the results there doesn't seem to be any obvious patterns here, most of the data is crowded around the low end of the playtime forever and no clear pattern with negative or positive reviews emerge. Next we compare playtime forever, and the playtime at review.

![Untitled](https://github.com/LM-UCSD/DSC232R-Spring-2024---Steam-Reviews/assets/128201733/856019d5-0f63-435b-9daa-32b08ad1e3d5)

There does seem to be a clear pattern here in the postiive direction between these two variables. It may make sense since the playtime forever is limiting the playtime at review, since it may never over take it. The edge is where the user stopped playing the game at the review. But there still isn't a clear indication that there is any patterns in when user's review positive or negative. Finally we compare the author's playtime in the last two weeks and author playtime at the review.

![Untitled](https://github.com/LM-UCSD/DSC232R-Spring-2024---Steam-Reviews/assets/128201733/0c37059f-f764-4214-a128-803ffea088c3)

This graph just looks like noise for the most part, if there is a patern betweeen the variables it is definitely not linear and not strong. There seems to be no pattern when comparing these two to each other. Seeing as there are no linear solutions that are obious to our questions. Our team seeked to use other means of predictive when users give positive or negative reviews, in order to make a prediction. The first choice could be a logistic regression as it can predict these binary outcomes, and the second choice could be a decision tree to visualize different decisions necessary in determining when a review will be positive. Thereout this paper, we seek to find a pattern that our initial data exploration does not make clear to us, although there are some catiousnary findings from the data exploration that we should keep in mind thereout, like the abundance of positive reviews that may bias the model towards a positive prediction.

Preprocessing

For our initial preprocessing of the data, we first enforced the data types manually as all the data types were initially loaded as strings. We then limited the reviews to English. This is not entirely necessary for our analysis that focused mostly on playtime; however, we wanted an English audience to understand the individual reviews if necessary. Most of the reviews are in English (>49%), but our results should be limited to English speakers reviewing games for this reason. Next we filtered the “voted_up” column to only include 0 (negative review) and 1 (positive review) . This was because there were anomalies in the data; for example, we found a review in this column in one of the cells. We then limited our analysis to the voted up, playtime in the last two weeks, playtime at review, and playtime overall as we are only studying this relationship and did not need any other columns as they were not relevant for our specific study. Next we then ensured that no null values were present in our data. Finally we created a VectorAssembler set to skip invalid cells for use in PySpark machine learning.

Model 1: Logistic Regression

The first implementatioin of our model uses logistic model. This was a logical choice as we are attempting to predict a binary variable of whether a user "upvotes" a particular game with their review based on various steam user metrics (Playtime in the last two weeks, at review, and overall). The ground truth of our model that we are assuming, is that as user play the game more (at review, overall, or over the last two weeks), we should expect that steam users would be more postive. Vice versa, if they play the game less we'd expect that we'd see the opposite.

Model 2: Decision tree

	Parameter Choices and rationale

	Model training and validation

Results

	Model 1: Logistic Regression Results
 
Our model achieves ~86% accurancy in our training set, and ~68% in our test set, this is likely a sign that our model is overfitting to our training set.
 
	Model 2: Decision Tree Results
	
		Performance/Accuracy
		
		Visualization of Model

Discussion Section

	Model 1: Logistic Regression Results

The first implementation of our model uses logistic regression. This was a logical choice as we are attempting to predict a binary variable of whether a user "upvotes" a particular game with their review based on various Steam user metrics (playtime in the last two weeks, at review, and overall). The ground truth assumption of our model is that as users play the game more (at review, overall, or over the last two weeks), we should expect that Steam users would be more positive in their reviews. Conversely, if they play the game less, we'd expect the opposite.

First, we used a model with a log loss metric and then an accuracy metric to evaluate the complexity of our model by examining the regularization parameter. In general, the model error rate sits ~39% for both test and train error in the log loss logistic model and stabilizes at around 13.58% for training error and 13.59% for test error in the accuracy logistic regression model. Which is not terrible performance, but we do have to keep in mind that much of the data is positive, and a postiive result always may be a better predictor than the model that gives both positive and negatives. With the logistic regression model, it seems in general however, our ground truth seems to be true, more playtime seeks increased postive reviews.

![Figure 7](https://github.com/LM-UCSD/DSC232R-Spring-2024---Steam-Reviews/assets/128201733/fb66d55c-6a79-4bf9-ae04-8abd12822f0e)

Our current model is fairly simple as it is a logistic regression model. In the log loss case, increasing the complexity (using a smaller regularization parameter) resulted in improvements to our test error, suggesting that we are likely to the left of the fitting graph and that complexity could improve our log loss model. 

![Figure 8](https://github.com/LM-UCSD/DSC232R-Spring-2024---Steam-Reviews/assets/128201733/70db058d-4da7-4ad9-a8d7-e3c1e2236ff)

However, in the case of the accuracy model, there seemed to be no improvement when increasing the size of the regularization parameter, as the error rates were stable and increased slightly in our smallest regularization parameter. In both cases our errors were fairly close to one another, suggesting we may be able to benefit from increased complexity in our models; although just a regularization parameter may not be the appropriate measure of complexity. 

We may want to consider the choice of our predictor variables, as many of them—such as playtime in the last two weeks and overall playtime—may be autocorrelated. One could argue that high playtime in the last two weeks is correlated with higher overall playtime; after all, as playtime in the last two weeks increases, overall playtime also increases. Therefore, we may want to consider selecting different or fewer variables that are not dependent on one another, as they could be causing additional error.

Overall, our model may be biased due to our dataset mostly containing positive reviews, we want to watch for overfitting; however, our logistic regression model performaned fairly well and seems to point towards increased playtime increasing the amount of postiive reviews like our ground truth.

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
Sahra Ranjbar: I worked on the fitting graph and running the models to ensure no need for editing. 


Final Model and Results and Summary

	Final Model

	Results Summary

