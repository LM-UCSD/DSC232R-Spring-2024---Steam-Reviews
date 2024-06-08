# DSC232R-Spring-2024---Steam-Reviews
Analysis of Steam user reviews

By Chase Farrell, Lauren Marrs, Alison Cher, Sahra Ranjbar

## 1.0 Introduction

Steam, a video game digital distribution service, hosts over 34,000 games and is one of the largest digital storefronts for video games. As part of their storefront, Steam hosts user reviews of the games they distribute. Users may leave a positive or negative review with text and other users may give feedback to that review whether it was helpful or otherwise. The portion of positive to negative reviews creates an overall metric and a recent metric ranging from Overwhelming Positive (95% over more positive), Very Positive (80% of more), Mostly Positive (70%-79%), Mixed (40%-69%), Mostly Negative (20%-39%), Overwhelmingly Negative (0%-19%). Generally, developers would prefer to have their games on the positive side as this should understandably increase sales overtime. This project attempts to provide insight into steam reviews to give developers a better understanding of why users leave negative or positive reviews. 

Using a 100+ Million Steam Review dataset sourced from Kaggle, our team seeks to better understand the relationship between steam players and their reviews. Using the data collected by the kaggle dataset, our team specifically explores the relationship between user playtime and whether they’d leave a negative or positive review. By using a logistic regression model, we attempt to predict whether a user leaves a positive or a negative review (Are we going to add a decision tree as well). The results of which allow developers to make decisions on the prioritization of user playtime in their games.

Kaggle link to our data:
https://www.kaggle.com/datasets/kieranpoc/steam-reviews/data 

The jupyter notebook contains code blocks to download the dataset.

Description from Kaggle:
- author
 - steamid
 - number of games owned
 - number of reviews
 - playtime all time
 - playtime over the last 2 weeks
 - playtime at the time of the review
 - when they last played the game
- language
- time created
- time updated
- if the review was positive or negative
- number of people who voted the review up
- number of people who voted the review funny
- a helpfulness score (steam generated)
- number of comments
- if the user purchased the game on Steam
- if the user checked a box saying they got the app for free
- if the user posted this review while the game was in Early Access
- developer response (if any)
- when the developer responded (if applicable)

When loading a SDSC Jupyter session we used:

- 4 cores
- 16 GB per node

## 2.0 Methods

### 2.1 Data Exploration

Techniques we used to explore our data include:
- Reviewing data types using `.dtypes`
- Summary statistics using `.describe()`
- Histogram visualization using `sns.histplot`
- Identifying basic trends between features using a Pearson Correlation Matrix

### 2.2 Preprocessing

Preprocessing techniques:
- Re-casted data types from strings to appropriate types (integer, float, etc.)
- Filtered reviews to english
- Filtered voted_up to 0 and 1
- Selected specific columns of interest
- Removed null values using `.na.drop`

To re-cast data types, first establish a list of `[('feature name', 'data type')]` and then:
```
for pair in types:
    column_name = pair[0]
    data_type = pair[1]
    Steam_data = Steam_data.withColumn(column_name, col(column_name).cast(data_type))
```

To select certain features of interest:
```
Select_Steam = Steam_data.select(
    'author_playtime_forever',
    'author_playtime_at_review',
    'author_playtime_last_two_weeks',
    'voted_up'
).cache()
```

### 2.3 Model 1: Logistic Regression

- Train Test Split: Random split 80/20
- Parametrization: Evaluated at increasing complexity (0.01, 0.1, 1, 10, 100)
- Evaluator: Multiclass Classification Evaluator with Log Loss and Accuracy metrics

Model setup:

```
train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)
log_reg = LogisticRegression(featuresCol="features", labelCol="voted_up")
log_reg_model = log_reg.fit(train_data)
param = [0.01, 0.1, 1, 10, 100]
```

Data generation:
```
train_error=[]
test_error=[]

evaluator = MulticlassClassificationEvaluator(labelCol='voted_up', predictionCol='prediction', metricName='logLoss')

for i in param:
    log_reg = LogisticRegression(featuresCol="features", labelCol="voted_up", regParam=1/i)
    log_reg_model = log_reg.fit(train_data)
    test_predictions = log_reg_model.transform(test_data)
    train_predictions = log_reg_model.transform(train_data)
    test_errors=evaluator.evaluate(test_predictions)
    train_errors=evaluator.evaluate(train_predictions)
    test_error.append(test_errors)
    train_error.append(train_errors)
```

### 2.4 Model 2: Decision Tree

- Train Test Split: Random split 80/20
- Evaluator: Multiclass Classification Evaluator with Accuracy metric

Model setup:

```
train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)
dec_tree=DecisionTreeClassifier(featuresCol="features", labelCol="label")
decision_tree=dec_tree.fit(train_data)0]
```

Data generation:
```
predict=decision_tree.transform(test_data)
predict.select("label", "prediction", "probability").show()
```

## 3.0 Results

### 3.1 Data Exploration Results

To explore the data, we first checked the data type of all the features. We converted the values to integers, as the whole table was stored as strings. From here, we selected features of interest and reviewed a few lines of the data table to understand the scope of the contents. After this, we pulled a .describe() command to see summary statistics of the data as well as a count of rows. 

+-------+-----------------------+-------------------------+------------------------------+------------------+
|summary|author_playtime_forever|author_playtime_at_review|author_playtime_last_two_weeks|          voted_up|
+-------+-----------------------+-------------------------+------------------------------+------------------+
|  count|               49791098|                 49791098|                      49791098|          49791098|
|   mean|     14778.687062092906|        6978.960364822644|             78.91529915648778|0.8665799657601445|
| stddev|     47516.476050580975|       26057.819471451778|             586.4386450258011|0.3400281326969518|
|    min|                    0.0|                        0|                           0.0|               0.0|
|    max|              6007985.0|                  4880175|                       56748.0|               1.0|
+-------+-----------------------+-------------------------+------------------------------+------------------+

We took a random .5% sample of the >100 million reviews, resulting in a 200,000 row sample to work with. We put this sub-sample into a pandas dataframe to visualize. 

Next we generated figures of our desired columns of study. We went through each temporal measurement of playtime (Playtime Forever, Playtime Last Two Weeks, and Playtime at review), with whether it was upvoted or not also shown. First the play time forever:

![Figure 1](https://github.com/LM-UCSD/DSC232R-Spring-2024---Steam-Reviews/assets/128201733/4153bdbc-4973-4567-a838-03330677b9d7)

This figure shows a relatively normal shape, with the majority of reviews being positive. The negative reviews seem to follow an approximately normal shape as well. In General there seems to be more reviews with more playtime, but low playtime games tend to be more negative. Other things this may show us is that there tends to be more positive reviews in our dataset. Next we show this for the Last Two weeks

![Figure 2](https://github.com/LM-UCSD/DSC232R-Spring-2024---Steam-Reviews/assets/128201733/087e0adb-ee40-4dfd-9d5a-8bb12a7c9ecc)
8)

Overall, this shape is different than the Playtime forever, there seems to be more negative reviews as playtime increase, but there is a peak at the low end of the graph. Perhaps recent playtime is not attributal to more reviews.

![Figure 3](https://github.com/LM-UCSD/DSC232R-Spring-2024---Steam-Reviews/assets/128201733/26560247-06ee-4271-8eff-6f128d4f5ebb)

Next we explore playtime at review and is seems to be a similar distribution as the playtime forever graph, this may suggest these two are related to one another. 

To futher our exploration, we do want to look for relationships with upvotes and downvotes and see if the playtimes are related to one another. Below we pair Playtime Forever and Playtime in the last two weeks together.

![Figure 4](https://github.com/LM-UCSD/DSC232R-Spring-2024---Steam-Reviews/assets/128201733/44844c6f-58c0-40a0-94b1-662f1c222483)

Looking at the results there doesn't seem to be any obvious patterns here, most of the data is crowded around the low end of the playtime forever and no clear pattern with negative or positive reviews emerge. Next we compare playtime forever, and the playtime at review.

![Figure 5](https://github.com/LM-UCSD/DSC232R-Spring-2024---Steam-Reviews/assets/128201733/856019d5-0f63-435b-9daa-32b08ad1e3d5)

There does seem to be a clear pattern here in the postiive direction between these two variables. It may make sense since the playtime forever is limiting the playtime at review, since it may never over take it. The edge is where the user stopped playing the game at the review. But there still isn't a clear indication that there is any patterns in when user's review positive or negative. Finally we compare the author's playtime in the last two weeks and author playtime at the review.

![Figure 6](https://github.com/LM-UCSD/DSC232R-Spring-2024---Steam-Reviews/assets/128201733/0c37059f-f764-4214-a128-803ffea088c3)

This graph just looks like noise for the most part, if there is a patern betweeen the variables it is definitely not linear and not strong. There seems to be no pattern when comparing these two to each other. Seeing as there are no linear solutions that are obious to our questions. Our team seeked to use other means of predictive when users give positive or negative reviews, in order to make a prediction. The first choice could be a logistic regression as it can predict these binary outcomes, and the second choice could be a decision tree to visualize different decisions necessary in determining when a review will be positive. Thereout this paper, we seek to find a pattern that our initial data exploration does not make clear to us, although there are some catiousnary findings from the data exploration that we should keep in mind thereout, like the abundance of positive reviews that may bias the model towards a positive prediction.

Finally, a Pearson Correlation Matrix was constructed to validate our selection:

```
Features across/down: 'author_playtime_forever','author_playtime_at_review','author_playtime_last_two_weeks','author_num_games_owned','author_num_reviews','voted_up'

DenseMatrix([[ 1.        ,  0.7886514 ,  0.34475927, -0.0321441 , -0.03143936, 0.0222244 ],
             [ 0.7886514 ,  1.        ,  0.20948163, -0.02643021, -0.02726179, -0.00461497],
             [ 0.34475927,  0.20948163,  1.        , -0.01921135, -0.01409563, 0.01132714],
             [-0.0321441 , -0.02643021, -0.01921135,  1.        ,  0.27862913, -0.03877855],
             [-0.03143936, -0.02726179, -0.01409563,  0.27862913,  1.        , -0.03944351],
             [ 0.0222244 , -0.00461497,  0.01132714, -0.03877855, -0.03944351, 1.        ]])
```
### 3.2 Preprocessing Results

For our initial preprocessing of the data, we first enforced the data types manually as all the data types were initially loaded as strings. We then limited the reviews to English. This is not entirely necessary for our analysis that focused mostly on playtime; however, we wanted an English audience to understand the individual reviews if necessary. Most of the reviews are in English (>49%), but our results should be limited to English speakers reviewing games for this reason. Next we filtered the “voted_up” column to only include 0 (negative review) and 1 (positive review) . This was because there were anomalies in the data; for example, we found a review in this column in one of the cells. We then limited our analysis to the voted up, playtime in the last two weeks, playtime at review, and playtime overall as we are only studying this relationship and did not need any other columns as they were not relevant for our specific study. Next we then ensured that no null values were present in our data. Finally we created a VectorAssembler set to skip invalid cells for use in PySpark machine learning.

### 3.3 Model 1 Results: Logistic Regression

The first implementatioin of our model uses logistic model. This was a logical choice as we are attempting to predict a binary variable of whether a user "upvotes" a particular game with their review based on various steam user metrics (Playtime in the last two weeks, at review, and overall). The ground truth of our model that we are assuming, is that as user play the game more (at review, overall, or over the last two weeks), we should expect that steam users would be more postive. Vice versa, if they play the game less we'd expect that we'd see the opposite.

First, we used a model with a log loss metric and then an accuracy metric to evaluate the complexity of our model by examining the regularization parameter. In general, the model error rate sits ~39% for both test and train error in the log loss logistic model and stabilizes at around 13.58% for training error and 13.59% for test error in the accuracy logistic regression model. Which is not terrible performance, but we do have to keep in mind that much of the data is positive, and a postiive result always may be a better predictor than the model that gives both positive and negatives. With the logistic regression model, it seems in general however, our ground truth seems to be true, more playtime seeks increased postive reviews.

![Figure 7](https://github.com/LM-UCSD/DSC232R-Spring-2024---Steam-Reviews/assets/128201733/fb66d55c-6a79-4bf9-ae04-8abd12822f0e)

Our current model is fairly simple as it is a logistic regression model. In the log loss case, increasing the complexity (using a smaller regularization parameter) resulted in improvements to our test error, suggesting that we are likely to the left of the fitting graph and that complexity could improve our log loss model. 

![Figure 8](https://github.com/LM-UCSD/DSC232R-Spring-2024---Steam-Reviews/assets/163374052/ae50cf86-1e09-42a4-9f41-0ab41bd34397)


Our model achieves ~86% accurancy in our training set, and ~68% in our test set, this is likely a sign that our model is overfitting to our training set.

### 3.4 Model 2 Results: Decision tree

The decision tree classifier resulted in 86.67% accuracy.

Sample predictions:

+-----+----------+--------------------+
|label|prediction|         probability|
+-----+----------+--------------------+
|  0.0|       1.0|[0.35060548180689...|
|  1.0|       1.0|[0.35060548180689...|
|  1.0|       1.0|[0.35060548180689...|
|  0.0|       1.0|[0.35060548180689...|
|  1.0|       1.0|[0.35060548180689...|
|  0.0|       1.0|[0.29483476944470...|
|  1.0|       1.0|[0.29483476944470...|
|  1.0|       1.0|[0.10309560043470...|
|  1.0|       1.0|[0.10309560043470...|
+-----+----------+--------------------+

Summary of prediction results:

+-----+----------+-------+
|label|prediction|  count|
+-----+----------+-------+
|  1.0|       0.0|  77119|
|  0.0|       0.0|  78019|
|  1.0|       1.0|8553009|
|  0.0|       1.0|1250902|
+-----+----------+-------+

## 4.0 Discussion 

### 4.1 Data Exploration

### 4.2 Preprocessing

### 4.3 Model 1: Logistic Regression

The first implementation of our model uses logistic regression. This was a logical choice as we are attempting to predict a binary variable of whether a user "upvotes" a particular game with their review based on various Steam user metrics (playtime in the last two weeks, at review, and overall). The ground truth assumption of our model is that as users play the game more (at review, overall, or over the last two weeks), we should expect that Steam users would be more positive in their reviews. Conversely, if they play the game less, we'd expect the opposite.

However, in the case of the accuracy model, there seemed to be no improvement when increasing the size of the regularization parameter, as the error rates were stable and increased slightly in our smallest regularization parameter. In both cases our errors were fairly close to one another, suggesting we may be able to benefit from increased complexity in our models; although just a regularization parameter may not be the appropriate measure of complexity. 

We may want to consider the choice of our predictor variables, as many of them—such as playtime in the last two weeks and overall playtime—may be autocorrelated. One could argue that high playtime in the last two weeks is correlated with higher overall playtime; after all, as playtime in the last two weeks increases, overall playtime also increases. Therefore, we may want to consider selecting different or fewer variables that are not dependent on one another, as they could be causing additional error.

Overall, our model may be biased due to our dataset mostly containing positive reviews, we want to watch for overfitting; however, our logistic regression model performaned fairly well and seems to point towards increased playtime increasing the amount of postiive reviews like our ground truth.

### 4.4 Model 2: Decision Tree

The second moodel we chose to use was a decision tree. This seemed like a reasonable alternative due to the binary label and straight forward features used.

While the model resulted in decent overall accuracy at 87%, this is a somewhat misleading metric. In reality, the decision tree was good at predicting *up* votes, but not *down* votes. This is likely due to the discrepancy between up and down votes in the data set, with up votes being significantly more common (people like video games!).


Since this model may have some extensive over-fitting, and generally points to *any* playtime resulting in a postitive review, it would not be our model of choice compared to logistic regression as it would not be particularly helpful for game developers to understand their ratings.

		
## 5.0 Conclusion

	Summary

	Future Work
	
	Final Thoughts


## 6.0 Statement of Collaboration

Lauren Marrs:  Did final debugging and cleaning for code, contributed to the write-up

Sahra Ranjbar: Worked on the fitting graph and running the models to ensure no need for editing and contributed to the write-up.

Chase Farrell: Coded the preprocessing, contributed to debugging, and contributed to the write-up

Alison Cher: Contributed bulk of the code for the ML algorithms and models.


## 7.0 Final Model and Results and Summary

In summary, we would recommend pursuing a logistic regression model to understand the relationship between playtime and user reviews. This model resulted in 86% accuracy, and did not have as any issues with misclassifying negative versus positive reviews. Understanding how users interact with a game before deciding whether they like it can help game developers bridge the missing data gap between the vocal minority (reviewers) and the silent majority (players who do not leave reviews). This can go a long way toward understand why a game is getting the feedback it is, and whether the content of a game is good while the marketing may be lacking, or perhaps it is on the right track in improving but needs better implementation, or if a minority of players is particularly up in arms about a feature while the majority are still happily playing.

