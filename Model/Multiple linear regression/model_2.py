# Multiple Linear Regression
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
import seaborn as sns

# Importing the datasets
df = pd.read_csv('organized-dataset.csv')

scaler = StandardScaler()

X = df.iloc[:, [0, 3, 6]]
Y = df.iloc[:, [2]]

# Splitting the dataset into the Training set and Test set
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size=0.2, random_state=50)

X_Train = scaler.fit_transform(X_Train)
X_Test = scaler.transform(X_Test)

# Fitting the Multiple Linear Regression in the Training set
regressor = LinearRegression()
regressor.fit(X_Train, Y_Train)

# K-fold cross validation
scores = cross_val_score(regressor, X_Train, Y_Train, cv=5)
print('Cross-validation scores:', scores)
print('Mean score:', np.mean(scores))
print('Standard deviation:', np.std(scores))

# Predicting the Test set results

Y_Pred = regressor.predict(X_Test)

score=r2_score(Y_Test,Y_Pred)
mae = mean_absolute_error(Y_Test, Y_Pred)
mse = np.sqrt(mean_squared_error(Y_Test, Y_Pred))
print("R2 score:", score, "Mean absolute error:", mae, "Mean squared error:", mse)

# Print test and prediction scores
print(Y_Test)
print(Y_Pred)
df1 = pd.DataFrame(Y_Pred,columns =["Error percentage predicted"])
df1.to_csv("error percentage.csv", index = False)
Y_Test.to_csv("error_percentage_test.csv" , index = False)

while True:
    input_str = input("Enter comma-separated values for input features: ")
    input_list = input_str.split(",")

    # Convert input to numpy array and reshape it to match training data shape
    new_input = np.array(input_list).astype(np.float64)
    new_input = new_input.reshape(1, -1)

    # Make a prediction on the new input
    prediction = regressor.predict(new_input)

    # Print the predicted target variable value for the new input
    print("Predicted target variable value:", prediction[0])