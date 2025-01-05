# IndustriAI-Hackathon
```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Sample data - For illustration, replace with actual data
# Features could be climate risk, operational risk, financial risk, etc.
data = {
    'Climate_Risk': [7, 5, 8, 6, 9],
    'Operational_Risk': [6, 7, 5, 6, 4],
    'Regulatory_Risk': [8, 6, 7, 9, 5],
    'Market_Risk': [5, 4, 6, 7, 8],
    'Total_Risk': [6.5, 5.3, 6.5, 6.3, 6.0],  # This is the target value, the risk index.
}

# Convert the data into a DataFrame
df = pd.DataFrame(data)

# Define features and target variable
X = df[['Climate_Risk', 'Operational_Risk', 'Regulatory_Risk', 'Market_Risk']]
y = df['Total_Risk']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')

# Show the risk index predictions
print(f'Predicted Risk Index: {y_pred}')

```
