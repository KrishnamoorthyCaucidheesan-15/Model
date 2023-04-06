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

# Using concurrency,ratelimit and timeout to train and predict error percentage
X_2= df.iloc[:, [0, 3, 6]]
Y_2= df.iloc[:, [2]]

# Splitting the dataset into the Training set and Test set
X2_Train, X2_Test, Y2_Train, Y2_Test = train_test_split(X_2, Y_2, test_size=0.2, random_state=75)

# Normalising the dataset
X2_Train = scaler.fit_transform(X2_Train)
X2_Test = scaler.transform(X2_Test)

# Fitting the Random forest regression in the Training set
regressor = RandomForestRegressor(n_estimators=100, max_depth=10)
regressor.fit(X2_Train, Y2_Train)

# Saving in a pickle file
# pickle.dump(regressor, open('model_2.pkl','wb'))
# model = pickle.load(open("model_2.pkl","rb"))

# K-fold cross validation
scores = cross_val_score(regressor, X2_Train, Y2_Train, cv=5)
print('Cross-validation scores:', scores)
print('Mean score:', np.mean(scores))
print('Standard deviation:', np.std(scores))

# Predicting the Test set results
Y2_Pred = regressor.predict(X2_Test)

# Obtain evaluation scores
score=r2_score(Y2_Test,Y2_Pred)
mae = mean_absolute_error(Y2_Test, Y2_Pred)
mse = np.sqrt(mean_squared_error(Y2_Test, Y2_Pred))
print("r2 score", score, "mean absolute error :", mae, "mean squared error:", mse)

# Print test and prediction scores
df1 = pd.DataFrame(Y2_Pred,columns =["Error percentage predicted"])
df1.to_csv("d.csv", index = False)
Y2_Test.to_csv("e.csv" , index = False)

# Loop for real-time user input
while True:
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

