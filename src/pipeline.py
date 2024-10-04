import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# Load the data
df = pd.read_csv('tips.csv')

# Label encoding for categorical variables
lb = LabelEncoder()
df['sex'] = lb.fit_transform(df['sex'])
df['smoker'] = lb.fit_transform(df['smoker'])
df['day'] = lb.fit_transform(df['day'])
df['time'] = lb.fit_transform(df['time'])

# Split features and target
x = df.drop(columns=['total_bill'])
y = df['total_bill']

# Standard scaling
sc = StandardScaler()
x_sc = sc.fit_transform(x)

# Convert scaled features back to DataFrame for consistency
x_new = pd.DataFrame(x_sc, columns=x.columns)

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x_new, y, test_size=0.2, random_state=42)

# Linear regression model
lr = LinearRegression()
lr.fit(x_train, y_train)

# Predict on the test set
y_pred = lr.predict(x_test)

# Calculate R^2 score
r2 = r2_score(y_test, y_pred)
print(f"R^2 Score: {r2}")
