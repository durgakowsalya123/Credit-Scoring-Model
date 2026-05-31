import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load dataset
data = pd.read_csv("credit_data.csv")

# Display dataset
print("\nDataset:")
print(data)

# Features (Input)
X = data[['Income', 'Debt', 'Payment_History']]

# Target (Output)
y = data['Creditworthy']

# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Logistic Regression model
model = LogisticRegression()

# Train model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nModel Evaluation:")
print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)

# Predict a new customer
print("\nNew Customer Prediction")

income = float(input("Enter Income: "))
debt = float(input("Enter Debt: "))
payment_history = int(input("Payment History (1=Good, 0=Bad): "))

new_customer = [[income, debt, payment_history]]

prediction = model.predict(new_customer)

if prediction[0] == 1:
    print("\nResult: Customer is Creditworthy")
else:
    print("\nResult: Customer is NOT Creditworthy")