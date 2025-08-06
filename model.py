
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load the dataset
df = pd.read_csv('data/stock_data.csv')

# Basic feature engineering (you can expand this later)
df['Date'] = pd.to_datetime(df['Date'])
df['Day'] = df['Date'].dt.day
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year

# Drop columns that won't help
df.drop(['Date', 'Stock Symbol'], axis=1, inplace=True, errors='ignore')

# Define features and target
X = df.drop('Close', axis=1)
y = df['Close']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, 'models/stock_price_model.pkl')

print("✅ Model trained and saved as models/stock_price_model.pkl")
