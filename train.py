import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Fetch the California housing dataset
housing = fetch_california_housing()
X, y = housing.data, housing.target

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the model to a file using pickle
with open('house_price_model.pkl', 'wb') as file:
    pickle.dump(model, file)
