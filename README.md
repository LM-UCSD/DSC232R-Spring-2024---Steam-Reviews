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
