# IndustriAI-Hackathon
```
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Sample data (Risk Index and the corresponding features)
data = {
    'Risk_Index': [6.2, 6.5, 5.3, 6.0, 5.8],  # Known Risk Index
    'Solar_Radiation': [5, 6, 7, 5.5, 6.2],  # Solar radiation hours per day (simulated)
    'Regulatory_Risk': [6, 7, 5, 6, 4],      # Regulatory Risk (simulated)
    'Technology_Maturity': [7, 8, 6, 7, 8],  # Technology Maturity (simulated)
    'Geographical_Risk': [5, 4, 6, 5, 7],    # Geographical Risk (simulated)
}

# Convert data into DataFrame
df = pd.DataFrame(data)

# Separate features and target
X = df[['Solar_Radiation', 'Regulatory_Risk', 'Technology_Maturity', 'Geographical_Risk']]
y = df['Risk_Index']

# Train a Linear Regression model
model = LinearRegression()
model.fit(X, y)

# Now, given a new Risk Index, we want to predict the features
# Let's say we have a new Risk Index value and want to generate features for it

# For example, given a new risk index value:
new_risk_index = np.array([6.0]).reshape(-1, 1)  # New risk index value

# Use the trained model to predict the features (reverse-engineering from the target)
# Predicting for each feature:
predicted_features = model.predict(new_risk_index)

# Show predicted values for the features corresponding to the new risk index
print(f'Predicted Features for Risk Index {new_risk_index[0][0]}:')
print(f'Solar Radiation: {predicted_features[0][0]}')
print(f'Regulatory Risk: {predicted_features[0][1]}')
print(f'Technology Maturity: {predicted_features[0][2]}')
print(f'Geographical Risk: {predicted_features[0][3]}')

```
