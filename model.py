import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle


# Load CSV file
df = pd.read_csv('iris.csv')

# Split data into X and y
X = df.drop('Class', axis=1)
y = df['Class']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=50)

# # Scale data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

# Instantiate model
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

# Train model
model.fit(X_train, y_train)

# Make predictions
# y_pred = model.predict(X_test)

# Evaluate model
# accuracy = accuracy_score(y_test, y_pred)

# Save model

pickle.dump(model, open('model.pkl', 'wb'))


