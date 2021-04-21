import pandas as pd
from sklearn import preprocessing , model_selection , neighbors
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
#print(plt.style.available)

df_classification = pd.read_csv('datasets/heart_failure_clinical_records_dataset.csv')
df_regression = df_classification[['age','creatinine_phosphokinase' , 'ejection_fraction' , 'platelets' , 'serum_creatinine' , 'serum_sodium' , 'time']]

#pd.set_option('display.max_rows' , None)

X_classification = df_classification.drop(['DEATH_EVENT' , 'creatinine_phosphokinase' , 'ejection_fraction' , 'platelets'] , 1)
y_classification = df_classification['DEATH_EVENT']

X_regression = df_regression.drop('time' , 1)
y_regression = df_regression['time']

X = np.array(df_regression.drop('time' , 1))
X = preprocessing.scale(X_regression)
#print(X , X_regression)

def classification(X_classification , y_classification , knn_neighbours):
    X_train , X_test , y_train , y_test = model_selection.train_test_split(X_classification , y_classification , test_size = 0.2 )
    clf = neighbors.KNeighborsClassifier(n_neighbors=knn_neighbours)
    #print(X_train.shape , y_train.shape)
    clf = clf.fit(X_train , y_train)
    score = clf.score(X_test , y_test)
    return clf , score

def plotting_score_classification(start_neighbour , end_neighbour , iterations , mean_iteration):
    # THIS HOW I CALCULATED THE MEAN OF THE ACCURACY
    # THE mean_iteration PARAMETER WILL BE THE NUMBER OF HOW MANY TIMES YOU WOULD WANT TO TAKE THE MEAN OF YOUR DESIRED ITERATIONS OF YOUR NEIGHBOURS
    # THE start_neighbour PARAMETER IS THE STARTING NUMBER OF KNN NEIGHBOURS YOU WOULD WANT TO SEE THE ACCURACY OF
    # THE end_neighbour PARAMETER IS TILL HOW MANAY NUMBER OF KNN NEIGHBOURS YOU WANT TO SEE THE ACCURACY OF
    # FOR EXAMPLE :- IF YOU SPECIFY start_neighbour = 3, end_neghbour = 10, iterations = 100, mean_iteration = 10 THEN IT WILL SHOW YOU THE PLOTS OF KNN NEIGHBOURS FROM 3 TO 10 WITH EACH NEIGHBOUR GIVING US THE ACCURACY SCORE 100 TIMES AND SHOWs 10 MEAN VALUES OF THESE DIFFERENT ITERATIONS
    if start_neighbour >= end_neighbour:
        raise Exception('The argument start_neighbour < end_neighbour')
    accuracy = {}
    mean_value = {}
    for k in range(1,mean_iteration+1):
        mean_value[k] = []
        for i in range(start_neighbour,end_neighbour+1):
            accuracy[i] = []
            for j in range(1,iterations):
                _ , score = classification(X_classification , y_classification , knn_neighbours = i)
                accuracy[i].append(score)
            mean_value[k].append(np.mean(accuracy[i]))
    # PLOTTING STARTS FROM HERE OF 100 ITERATIONS.
    # NOTE :- THE PLOTTINGS OF THE accuracy HERE WILL BE OF THE LAST ITERATION i.e. THE end_neighbour TH ITERATION VALUES WILL BE THE VALUES PLOTTED HERE
    color = {1:'green' , 2:'blue' , 3:'red' , 4:'cyan' , 5:'magenta' , 6:'yellow' , 7:'black' , 8:'white'}
    for i in range(1,end_neighbour-start_neighbour+1):
        plt.subplot(2 , 3 ,i)
        plt.plot(range(1,101) , accuracy[2+i] , color = color[i] , label = 'Knn-{}'.format(2+i))
        plt.xticks(range(1,101))
        plt.xlabel('iterations')
        plt.ylabel('accuracy of the classifier')
    plt.show()

    # IN THIS PART I TOOK THE MEAN OF SPECIFIED iterations VALUE PARAMETER AND PRESENTED IT IN A MORE READABLE FORMAT 
    for j in range(end_neighbour-start_neighbour+1):
        print('The mean values of {} knn neighbours : '.format(start_neighbour+j) , end= ' ')
        for i in range(1,mean_iteration+1):
            print(end= '{} '.format(mean_value[i][j]))
        print()

def prediction_classification(X_classification , y_classification , prediction):
    classification_clf , score = classification(X_classification , y_classification , knn_neighbours = 9)
    prediction = prediction.reshape(len(prediction),-1)
    return classification_clf.predict(prediction) , score

def regression(X_regression , y_regression):
    #X_train , X_test , y_train , y_test = model_selection.train_test_split(X_regression , y_regression , test_size = 0.2)
    clf = LinearRegression()
    clf = clf.fit(X_regression , y_regression)
    #score = clf.score(X_test , y_test)
    return clf

def regression_plot(df_regression):
    columns = df_regression.columns
    color = {1:'green' , 2:'blue' , 3:'red' , 4:'cyan' , 5:'magenta' , 6:'yellow'}
    for i in range(1,len(columns)):
        plt.style.use(style_use[i-1])
        plt.subplot(2,3,i)
        plt.plot(y_regression , df_regression[columns[i-1]] , color = color[i] , alpha = 0.6)
        plt.title('{} vs time'.format(columns[i-1]))
        plt.xlabel('Independent Variables')
        plt.ylabel('Dependent Variable')
    plt.show()

def regression_predict(regression_clf , X):
    X = np.array(X).reshape(len(X),-1)
    prediction = regression_clf.predict(X)
    return prediction

classification_clf = classification(X_classification , y_classification , 5)
regression_clf = regression(np.array(X_regression['age']).reshape(-1,1) , y_regression)
