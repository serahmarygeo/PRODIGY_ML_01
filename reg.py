from sklearn.linear_model import LinearRegression
import numpy as np

# Generate random sample data
np.random.seed(0)  # Set random seed for reproducibility
num_samples = 1000  # Number of samples
min_square_footage = 800  # Minimum square footage
max_square_footage = 3000  # Maximum square footage
min_bedrooms = 1  # Minimum number of bedrooms
max_bedrooms = 6  # Maximum number of bedrooms
min_bathrooms = 1  # Minimum number of bathrooms
max_bathrooms = 4  # Maximum number of bathrooms

# Generate random values for square footage, bedrooms, and bathrooms
X = np.random.randint(min_square_footage, max_square_footage, size=(num_samples, 3))

# Random prices based on square footage, bedrooms, and bathrooms
coefficients = np.random.uniform(50000, 150000, size=3)  # Random coefficients for each feature
y = np.dot(X, coefficients) + np.random.normal(0, 50000, size=num_samples)  # Add random noise

# Create and fit the model
model = LinearRegression()
model.fit(X, y)

# Predict prices for new data
new_data = np.array([[1300, 2, 1.5],  # Example new data point 1
                     [1700, 3, 2]])   # Example new data point 2
predicted_prices = model.predict(new_data)

# Print detailed output
print("Sample Data:")
print("Square Footage | Bedrooms | Bathrooms | Actual Price")
for i in range(5):  # Print first 5 samples for demonstration
    print("{: <14} | {: <8} | {: <9} | {: <13}".format(X[i, 0], X[i, 1], X[i, 2], y[i]))

print("\nTrained Model:")
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

print("\nPredicted Prices for New Data:")
for i, price in enumerate(predicted_prices):
    print("New Data Point {}: ${:,.2f}".format(i+1, price))