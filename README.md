# Will a customer subscribe to the "Term Deposit"?

## Overview
This Portuguese banking dataset has been retrieved from the UC Irvine machine learning repository. 
https://archive.ics.uci.edu/dataset/222/bank+marketing

There are 2 different datasets in various sizes.
For the sake of processing, 4521 entries were selected. 
The goal is to accurately predict/classify whether a customer will subscribe to the "Term Deposit" product offered by the bank.

I am following the CRISP-DM Framework for this analysis. 

### 1 - Business Understanding
    1.1 - Business problem definition
        The goal of this problem is to develop a classification model that will correctly classify/predict a customer
        will subscribe to the "Term Deposit" product offered by the bank. 
        European banks are under a lot of pressure to increase financial assets due to competition and the worldwide financial crisis. 
        To overcome this, they offer a long-term deposit with a good interest rate, using directed marketing campaigns. 
        Directed marketing can be more efficient by targeting customers more likely to be interested in the product. 
        However, directed marketing can also lead to privacy concerns and a negative perception of banks. 
        To balance this, banks need to improve efficiency by reducing the number of contacts while maintaining a similar level of success (clients subscribing to deposits).

    1.2 - Data problem definition
        The data objective is to develop a model that accurately predicts/classifies if contact with a customer will lead to success.
        In other words, whether a client subscribes to the term deposit. 
        This model should also help identify key characteristics that influence success.

### 2 - Data Understanding
    2.1 Data Collection
        -   bank.csv dataset is present in the data folder.
        -   Data will be loaded with pd.read_csv() method of Pandas

    2.2 Data Description
        The cursory overview of the data
	    -	Total 4521 Samples and 17 Features
	    -	Categorical data - job, marital, education, default, housing, poutcome,
                                loan, contact, y, month
	    -	Numerical data - age, balance, day, duration, campaign, pdays, previous
        -   No missing or duplicate entries.
        -   "Unknown" categorical data found in job, poutcome, education and contact categorical columns.

    2.3 Data Exploration
        ⁃	Check the describe function to identify the mean and standard deviation
        -	Identified unique values for Categorical features
        -   Plotted class distribution of the target y, imbalanced class observed.
        -   Only a 12% success rate was observed.

    2.4 Data Visualization
        -   Heatmap of co-relations of numeric features.
                Duration has the highest positive correlation
                Pdays and previous are co-related to each other
        -   Pie chart of subscription distribution.
                Classes are highly imbalanced. This may affect the ROC-AUC and F1 score
        -   Bar plot of Subscribed customers based on job type
                Management job customers are highly likely to subscribe along with technicians/blue-collar jobs
        -   Bar plot and pie plot for marital status for customers subscribed
                Married customers are more likely to subscribe
        -   Bar plot and pie plot for education for customers subscribed
                The highest number of subscribers came from secondary education
        -   Bar plot and pie plot for contact for customers subscribed 
                Cellular is the predominant mode of contact
        -   Bar plot of Subscribed customers based on month
                Summer months tend to have better subscription results. Specifically May.
        -   Bar plot for default, loan, housing for customers subscribed
        -   pie plot for default, loan, housing  for customers subscribed
                99% of People with no credit default subscribed
                90% of people with no loan subscribed
                Housing doesn't have much effect on subscriptions.
        -   Histograms of numeric features Age
                Age range 30-40 tend to subscribe 
        -   Histogram of Balance distribution
                People with a balance of less than 150k tend to subscribe.

### 3 - Data Preparation
    3.1 Data Cleaning
        -   Drop the day of the month column that would not provide any information.

    3.2 Data Imputation
        -   Map categories with "yes" and "no" to 1 and 0 for default, housing, loan and even y fields.

### 4 - Modeling
    4.1 Data split
        -   train(70%) and test(30) with train_test_split with random_state=42
        -   Create a list for categorical data vs numeric data
    
    4.2 Preprocessors
        -   Created a numeric pipeline with StandardScaler
        -   Created a categorical pipeline with OneHotEncoder
        -   Combined both in preprocessor with ColumnTransformer

    4.2 Logistic Regression 
        -   Create a pipeline with preprocessor and LogisticRegressor with max_iter=5000
        -   Define a params_grid with C values as [0.001, 0.01, 0.1, 1, 10, 100]
        -   Run GridSearchCv with the pipeline and params_grid with scoring as ROC-AUC
        -   Fit the model on the training dataset
        -   Extract and plot the feature importance data 
        -   Store all performance params in resultDf

    4.3 K-Nearest Neighbors
        -   Create a pipeline with preprocessor and KNeighborsClassifier
        -   Define a params_grid with n_neighbors values as [3, 5, 7, 9]
        -   Run GridSearchCv with the pipeline and params_grid with scoring as ROC-AUC
        -   Fit the model on the training dataset
        -   Extract and plot the feature importance data 
        -   Store all performance params in resultDf

    4.4 Decision Tree Classifier
        -   Create a pipeline with preprocessor and DecisionTreeClassifier
        -   Define a params_grid with max_dept values as [3, 5, 7, 10]
        -   Run GridSearchCv with the pipeline and params_grid with scoring as ROC-AUC
        -   Fit the model on the training dataset
        -   Extract and plot the feature importance data 
        -   Store all performance params in resultDf

    4.5 Support Vector Machine
        -   Create a pipeline with preprocessor and SVC
        -   Define a params_grid with C values as [0.1, 1, 10], kernel linear, RBF
        -   Run GridSearchCv with the pipeline and params_grid with scoring as ROC-AUC
        -   Fit the model on the training dataset
        -   Extract and plot the feature importance data 
        -   Store all performance params in resultDf

    4.6 Analyze
        -   Calculate accuracy, recall, precision, 1-score, ROC-AUC.
        -   Identify the mean fit time
        -   Create a data frame to analyze the model performance parameters.
        -   Plot the graphs for permutation importance.
        -   Plot the confusion matrix on test data. visually analyze the model performance. 

### 5 - Evaluation
    5.1 Overall performance
        -   Model Performance
        <p align="center">
        <img src=“images/“crisp.png>
        </p>
        -   Average ROC-AUC values 0.62
        -   Accuracy acceptable at 89%
        -   F1-score and Recall have poor  results ranging in 0.3

    5.2 Iterative Improvements
        -   Based on the re-runs of the above changes to make
            -   Tune hyperparameters of the model based on the previous runs and scores.
            -   Acknowledge that the poor model performance is due to class imbalance, 
                either use the resampling technique to balance out the classes and rerun the models
                or use Better models that work great with imbalanced classes like Gradient Boost.

    5.3 Overall performance with SMOTE resampling
        -   Model Performance
        <p align="center">
            <img src=“images/“crisp.png>
        </p>
        -   We can see the difference in model performance. ROC-AUC values rose from 0.62 to 0.92
        -   Accuracy increased from 89% to 94%
        -   F1-score dramatic increase from 0.3 to 0.9
        -   Recall also increase from 0.28 to 0.9

    5.4 Model Performance and comparision
        -   Considering the F1-score and ROC-AUC, All 4 models performed well with very minor differences.
        -   Logistic Regression outperformed SVM in F1-score, ROC-AUC as well as mean fit time.
        -   SVM has the highest precision score and the worst fit time of 8 seconds
        -   Decision Tree Classifier has a recall score and is lowest on all other params
        -   K-Nearest Neighbors has the lowest fit time average scores ranging in 0.91

    Based on the comparison metrics and the requirements, the User should choose Logistic regression or SVM for their classification. 
    However, if the dataset is higher and not much computational power is available, then SVMs can lead to incorrect and incomplete results.

### 6 - Deployment
    6.1 Overview
        European banks are currently looking to optimize the way customers are targeted directly and produce better subscription rates. They would
        need to identify the success attributes of the last campaign, ideal the demographic to focus on, any seasonal parameter, and even the number of times
        a customer needs to be contacted to seal the deal. By analyzing the given dataset here are some insights into these factors and some suggestions for 
        further steps you can take to increase the success rate of customer acquisition and optimize the campaign to save on valuable resources and money.

    6.2 Strategies to implement
        1.  Targeted Marketing Campaigns: Identify a demographic that is likely subscribe and tailor marketing strategies around to increase conversion rates.
        2.  Invest in further data accumulation: Understand that since the conversion rate is only 11%, further data accumulation is needed. Apart from the features
            that we currently have try to get customers' life events, risk appetite, and even their financial habits.
        3.  Customer Service: Build trust with customers by providing transparent information about the term deposit product, understanding their concerns,
            provide incentives for subscribing to the term deposit. Provide excellent customer service for their issues and concerns. Follow up periodically.
        4.  Feedback/Complaints: Collect feedback/complaints from customers who have subscribed to a term deposit to understand their experience and improve future offerings.
        5.  Referral Programs: Advertise referral incentives to existing and new members for successful subscriptions.

    6.5 Summary
        By leveraging these insights and taking proactive steps to optimize your campaign strategies, you can enhance your competitiveness, 
        attract more customers, and maximize profitability in the banking sector along with improved customer satisfaction.
