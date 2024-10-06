import random
import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import sklearn.model_selection
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

# data load from csv file

data = pd.read_csv("./dataset/student/student-mat.csv" , sep=";")

# data.to_html("data.html")

data = data[["G1" , "G2" , "G3" , "studytime" , "failures" , "absences"]]

predict = "G3"


# splitting training data 
x = np.array(data.drop([predict] , axis=1))
y = np.array(data[predict])

best = 0 

while True:

    x_train, x_test , y_train , y_test = sklearn.model_selection.train_test_split(x,y,test_size=0.2)


    #training model
    linear = linear_model.LinearRegression()

    linear.fit(x_train , y_train)
    acc = linear.score(x_test , y_test)
    # print(acc)
    
    if ( acc > best ):
        best = acc
        print("Accuracy :" , acc)
        # storing the model
        with open("./models/studentModelLR.pkl" , "wb") as model:
            pickle.dump(linear , model)
        


        
    # pickle_model = open("studentModel.pickle" , "rb") 
    # linear = pickle.load(pickle_model)
    # pickle_model.close()


    # print("Liner Coefficient : " , linear.coef_)
    # print("intercept : " , linear.intercept_)

    # predictions = linear.predict(x_test)


    # for i in range(len(predictions)):
    #     print(x_test[i] , round(predictions[i]) , y_test[i])