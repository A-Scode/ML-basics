from email import header
import random
from matplotlib import pyplot, table
import pandas as pd
import numpy as np
import  sklearn.model_selection
import pickle

import sklearn.model_selection.tests
from tabulate import tabulate

data = pd.read_csv("./dataset/student/student-mat.csv" , sep=";")

data = data[['G1', 'G2' , 'G3' , 'studytime' , 'failures' , 'absences']]

predict = 'G3'

# splitting data
x = np.array(data.drop([predict] , axis=1))
y = np.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.5, random_state=random.randint(1,100))



model_file = open("./models/studentModelLR.pkl", "rb")
linear = pickle.load(model_file)
model_file.close()

# predicting the output
y_pred = linear.predict(x_test)


print(tabulate( [[round(y_pred[i]), y_test[i]] for i in range(len(y_pred))] , headers=["Predicted G3", "Actual G3"] , tablefmt="grid"))

count_match = 0
for i in range(len(y_pred)):
    if(round(y_pred[i]) == y_test[i]):count_match+=1

print("Match Count: ", count_match , '/', len(y_pred))
print("Linear Coefficients: " , linear.coef_)
print("Linear Intercept: " , linear.intercept_)


# plotting line of regression

x_line = np.linspace(0, 20, 200)
y_line = linear.coef_[2]*x_line + linear.intercept_


pyplot.style.use('seaborn-v0_8')
pyplot.scatter([x_test[i][0] for i in range(len(x_test))], y_pred)
pyplot.xlabel('G1')
pyplot.ylabel('Final Grade')

# line of regression
# pyplot.plot(x_line,y_line , color='red', label='Regression Line')
pyplot.legend()
pyplot.show()