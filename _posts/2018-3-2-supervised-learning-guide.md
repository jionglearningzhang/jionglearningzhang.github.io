---
layout: post
title: General guide of supervised learning data science project
---

>This writing works as a reminder of the supervised learning workflow for Data Science projects. Follow the steps listed.


# 1. Exploratory Data Analysis (EDA)
Once get the data, try to understand the general goal and the data. EDA is an approach to analyze data sets and summarize their main characteristics, often with plots. EDA helps data scientists get a better understanding of the dataset at hands, and guide them to preprocess data and engineer features effectively.

1. Look at examples.
2. Look at distributions (means, median, standard deviation, histograms, CDFs, Q-Q plots, etc.).
3. Consider the outliers
4. Report noise/confidence.
5. Slice your data, look at your data in groups.
6. Check for consistency over time.


# 2. Problem Definition
Understand your business problem and the data background very well and then abstract and formulate it to a machine learning problem. The outcome of the machine learning should be able to directly fit in the A/B testing. The outcome should be reasonably actionable in the real world scenario. Keep in mind that we either want to make money or make our customers happy or both. A good problem definition will also help you to get good machine learning results.

 1. How will you use this model in real world business scenario?
 2. What are the base samples you will apply your machine learning model on?
 3. What is used as the label?
 4. What metrics do you care about?
 5. Is the dataset imbalanced? How?


# 3. Literature Review
Try to look for similar projects people have workd on to avoid reinventing the wheels and get inspired.

1. Google (papers, blogs, wiki ...) ; )
2. The official Kaggle blog, artciles under category Tutorials and Winner’s Interviews.
3. Kaggle competitions similar projects.
4. Tech startups' engineering blogs.


# 4. Evaluation Framework (pipeline) Establishment

From your problem definition, you should have a good idea of how your machine learning framework looks like.

1. Decide the metrics of the model you will use: accuracy, precision, recall, F1, AUC ...
2. Start with 2 simple features to build up your framework.
3. Get the cross-validation codes in the framework ready.
4. The inputs for the framework is X, y, model, parameters.


# 5. Feature Engineering

“Coming up with features is difficult, time-consuming, requires expert knowledge. ‘Applied machine learning’ is basically feature engineering.” – Andrew Ng

While in deep learning we usually just normalize the data (e.g., such that image pixels have zero mean and unit variance), in traditional machine learning we need handcrafted features to build accurate models. Doing feature engineering is both art and science, and requires iterative experiments and domain knowledge. Feature engineering boils down to feature selection and creation.

Feature creation
1. Talk to domain experts to get inspirations (some KPI can be used as baseline).
2. Seperate date into year, month, day, weekday or weekend, etc.
3. Replace zipcode with coordinates and other area info.
4. Aggregation, Percentage, intervals
5. Interactions between existing features.
6. Unsupervised learning

Feature Selection
    Features can be selected through two routes: from feature analysis and from model performance.

 1. Feature analysis  
        a. Feature distribution on labels.  
        b. Feature attributes (ex. stats, boxplot, correlations, ...).
 2. Feature selection  
        a. Removing features with low variance  
        b. Univariate feature selection  
        c. Recursive feature elimination  
        d. Selecting from model  


​		
# 6. Model / Parameters Selection

Given the input features' data structure, think about which model can be a good fit. For these models, understand how its parameters will influence the results. Utilize your established pipeline to do grid search and select the best model.

Iterate step 5, 6.


# 7. Ensemble
Use stacking to put different models together.

# 8. Model Production
1. Review and clean up.
2. Prepare the clean pipeline in codes.
3. Prepare the trained production model in storage (ex. pickle, json).
4. Refine all the documentation into a story.


Remember: Clearly documented, reproducible codes and data files are crucial.

