import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)


print("\ncleaned_data")
cleaned_data = pd.read_csv('Cleaned_data_2019.csv')
for column in cleaned_data.columns:
    print(column, set(cleaned_data[column]))
print('cleaned_data: ',cleaned_data.shape)
# Chronic_Pain {0, 1}
# High_impact_chronic_pain {0, 1}
cleaned_data.drop(['High_impact_chronic_pain'], axis=1, inplace=True)

# Modeling
print("\nModeling")
X = cleaned_data.drop('Chronic_Pain', axis=1)  # Features
y = cleaned_data['Chronic_Pain']  # Target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=1000,random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(report)