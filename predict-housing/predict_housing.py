import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the data
df = pd.read_csv('housing.csv')
print("First 5 rows:\n", df.head())

# Data cleaning - drop missing values
df = df.dropna() # drop NaN values

# Encode categorical variables (example: Location)
df = pd.get_dummies(df, columns=['Location'], drop_first=True)

# Visualise data
sns.pairplot(df)
plt.show()

sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.show()

# Split data
X = df.drop('Price', axis=1)
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse}")
print(f"R² Score: {r2}")

# Visualise predictions
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linewidth=2)

plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")
plt.show()

# Predict price for a new house (optional)
new_house = pd.DataFrame({
    'Size': [1800],
    'Bedrooms': [3],
    'Bathrooms': [2],
    'Age': 10,
    'Location_City': [1] # 1=City 0=Suburb
})

# Ensure columns match training data
new_house = new_house.reindex(columns=X_train.columns, fill_value=0)

predicted_price = model.predict(new_house)
print("\n<------------------------------>")
print(f"Predicted price of the new house: £{predicted_price[0]:,.2f}")
print("<------------------------------>\n")