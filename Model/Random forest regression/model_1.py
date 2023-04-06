# Import libraries
import numpy as np
import pandas as pd
from sklearn.metrics import  r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import pickle

# Importing the datasets
df = pd.read_csv('organized-dataset.csv')

# Scalar function
scaler = StandardScaler()

# Using concurrency and timeout to train and predict latency
X_1 = df.iloc[:, [0, 6]]
Y_1 = df.iloc[:, [5]]
print(X_1)
print(Y_1)

# Splitting the dataset into the Training set and Test set
X1_Train, X1_Test, Y1_Train, Y1_Test = train_test_split(X_1, Y_1, test_size=0.2, random_state=50)

# Normalising the dataset
X1_Train = scaler.fit_transform(X1_Train)
X1_Test = scaler.transform(X1_Test)

# Fitting the Random forest regression in the Training set
regressor = RandomForestRegressor(n_estimators=100, max_depth=5)
regressor.fit(X1_Train, Y1_Train)

# Saving in a pickle file
# pickle.dump(regressor, open('model_1.pkl','wb'))
# model = pickle.load(open("model_1.pkl","rb"))

# K-fold cross validation
scores = cross_val_score(regressor, X1_Train, Y1_Train, cv=5)
print('Cross-validation scores:', scores)
print('Mean score:', np.mean(scores))
print('Standard deviation:', np.std(scores))

# Predicting the test set results
Y1_Pred = regressor.predict(X1_Test)

# Obtain evaluation scores
score=r2_score(Y1_Test,Y1_Pred)
mae = mean_absolute_error(Y1_Test, Y1_Pred)
mse = np.sqrt(mean_squared_error(Y1_Test, Y1_Pred))
print("R2 score:", score, "Mean absolute error:", mae, "Mean squared error:", mse)

# Print test and prediction scores
print(Y1_Test)
print(Y1_Pred)
df1 = pd.DataFrame(Y1_Pred,columns =["Latency predicted"])
df1.to_csv("d_1.csv", index = False)
Y1_Test.to_csv("e_1.csv" , index = False)

# Loop for real-time user input
# while True:
    # Get user input
input_str = input("Enter comma-separated values for input features: ")
input_list = input_str.split(",")

# Convert input to numpy array and reshape it to match training data shape
new_input = np.array(input_list).astype(np.float64)
new_input = new_input.reshape(1, -1)

# Make a prediction on the new input
prediction = regressor.predict(new_input)

# Print the predicted target variable value for the new input
print("Predicted target variable value:", prediction[0])
