#Importing modules and libralies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


#Reading datasheet datas 
datas = pd.read_csv('Real-estate-data.csv')
table = datas.head()
maths = datas.describe()


#Short summury of contents in tabular form
print("\n\nThe following are the data statistical table:-\n\n")
print(table)


#Datas statistical calculations
print("\n\nThe folowing are the data mathematical calulations:-\n\n")
print(maths)


#Datasheet informations summury
print("\n\nThe informations concerning the datas are:-\n\n")
data_info = datas.info()
print(data_info)


#The shape of the dataframe
print("\n\nThe shape of the dataframes is:\n\n")
frame_shape = datas.shape
print(frame_shape)



#The number of variables in each datasheet
print("\n\nThe following are the number of variable in each data column:\n\n")
data_count = datas.count
print(data_count)


#Datatype of each column
print("\n\nThe datatype of each column of the datasheet is:\n\n")
data_type = datas.dtypes
print(data_type)



#Value of X to be plotted on the data training graph
print("\n\nThe graphical array values of X are:\n\n")
x = datas[['X1 transaction date','X2 house age','X3 distance to the nearest MRT station','X4 number of convenience stores','X5 latitude','X6 longitude']].values
print(x)



#Value of Y to be plotted on the data training graph
print("\n\nThe graphical array values of Y are:\n\n")
y = datas[['Y house price of unit area']].values
print(y)



#The array data shape of X datas
print("\n\nThe following is the array data shape of X values:\n")
x_shape = x.shape
print(x_shape)



#The array data shape of X datas
print("\n\nThe following is the array data shape of y values:\n")
y_shape = y.shape
print(y_shape)



#Making a model and determining an intercept and coefficient/gradient from the model's best fit line
model = LinearRegression()
model.fit(x,y)
coefficient = model.coef_
intercept = model.intercept_
print("\n\nFrom the best fit,the coefficient of the graph is:",coefficient)
print("\nFrom the best fit,the intercept of the graph is:",intercept)



#Model training prediction
model_prediction = model.predict(x)
print("\n\nFrom the graphs model best fit, the price data prediction would be:\n\n")
print(model_prediction)



#Mean Square Error of the model prediction
print("\n\nThe mean squared error of the the model is:-")
mse = mean_squared_error(y,model_prediction)
sqr_mse = np.sqrt(mse)
print("{:.4f}".format(sqr_mse))


#R2 score of the model prediction
print("\n\nThe model r2 score is:-")
model_r2_score = r2_score(y,model_prediction)
print("{:.4f}".format(model_r2_score))



#Set Score of the model prediction
print("\n\nThe model trainig data set score is:-")
model_train_score = model.score(x,y)
print("{:.4f}".format(model_train_score))


#Model and dataframe statistical and linear regression interpreted values table
print("\n\nThe following is the model statistical sumary table:")
model = sm.OLS(y, x).fit()
model.summary()
