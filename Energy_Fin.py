import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import csv
from sklearn.linear_model import LinearRegression, HuberRegressor 
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeRegressor

df = pd.read_csv('Elec_consumption.csv', sep=";")

df.rename(columns={'Unnamed: 0':'Year'}, inplace=True)

cons = df['Construction'].to_numpy()
tot = df['Total consumption'].to_numpy()
part = list(cons/tot)
tot = list(tot)
remov = []
remov1 = []
for item in part:
    if item > 0.0082:
        i = part.index(item)
        remov.append(tot[i])
        remov1.append(item)

for item in remov:
    tot.remove(item)
for item in remov1:
    part.remove(item)
        
X = np.array(tot).reshape(-1,1)
y = np.array(part)*100

plt.scatter(X,y)
plt.show()

degrees = [4, 6, 8]    
tr_errors = []   
val_errors = []       

X_train, X_val, y_train, y_val= train_test_split(
    X,y, test_size=0.5, train_size=None, random_state=42, shuffle=True, stratify=None) 

for i in range(len(degrees)):  
    poly = PolynomialFeatures(degree=degrees[i])
    X_train_poly = poly.fit_transform(X_train)
    
    lin_regr = LinearRegression(fit_intercept=False)
    lin_regr.fit(X_train_poly, y_train)
        
    y_pred_train = lin_regr.predict(X_train_poly)
    tr_error = mean_squared_error(y_train, y_pred_train)
    X_val_poly = poly.fit_transform(X_val)
    y_pred_val = lin_regr.predict(X_val_poly)
    val_error = mean_squared_error(y_val,y_pred_val)
       
       
    print("The first two row of X_poly: \n",X[0:2])
        
    print("\nThe learned wegihts: \n",lin_regr.coef_)
        
    tr_errors.append(tr_error)
    val_errors.append(val_error)
    
    X_fit = np.linspace(5000, 95000, 1000)
    plt.plot(X_fit, lin_regr.predict(poly.transform(X_fit.reshape(-1, 1))), label=("Model Training, polynomial degree = ", degrees[i]))
    plt.scatter(X_train, y_train, color="b", s=10, label="datapoints from the training dataframe" ) 
    plt.scatter(X_val, y_val, color="r", s=10, label="datapoints from the validation dataframe" )    # plot a scatter plot of y(maxtmp) vs. X(mintmp) with color 'blue' and size '10'
    plt.xlabel('Total consumption')    # set the label for the x/y-axis
    plt.ylabel('Construction part (%)')
    plt.legend(loc="best")    # set the location of the legend
    plt.title('Construction part of total energy consumption')    # set the title
    plt.show()    # show the plot
        
    
    
    
    
    