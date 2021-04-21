# Basic-KNN-and-Linear-Regression

I have added two basic models of machine learning i.e. ** K nearest neighbour ** and ** Linear Regression **.
1.  I took this [Heart Failure Clinical Records](datasets/heart_failure_clinical_records_dataset.csv) dataset from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets.php) which has many datasets.  
2.  In K nearest neighbour model, i did some analysis in what features to be considered which would give us the best accuracy in making predictions.
3.  I also did some analysis on how many neighbors would be efficient and good in prediction. Ps. I found that keeping the parameter **9** will give us better accuracy for this dataset.
4.  In linear regression model, i analyze that which features would be good for the model, so it turned out that some features were not linear with my target variable so i dropped those features and did my prediction.
5.  You will find out that there are some plotting functions in [models](model.py) file to visualize the data given and make the best out of it.
