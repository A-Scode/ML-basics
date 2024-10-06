from email import header
from matplotlib import table
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
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2, random_state=42)

model_file = open("./models/studentModelLR.pkl", "rb")
linear = pickle.load(model_file)
model_file.close()

# predicting the output
y_pred = linear.predict(x_test)

print("Linear Coefficients: " , linear.coef_)
print("Linear Intercept: " , linear.intercept_)

print(tabulate( [[round(y_pred[i]), y_test[i]] for i in range(len(y_pred))] , headers=["Predicted G3", "Actual G3"] , tablefmt="grid"))

# for i in range(len(y_pred)):
#     print("Predicted G3: ", y_pred[i], "Actual G3: ", y_test[i])




