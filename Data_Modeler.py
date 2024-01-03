import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
# import pycaret as pc
import tpot
from tpot import TPOTClassifier


print("\ncleaned_data")
cleaned_data = pd.read_csv('Cleaned_data_2019.csv')
for column in cleaned_data.columns:
    print(column, set(cleaned_data[column]))
print('cleaned_data: ',cleaned_data.shape)
# Chronic_Pain {0, 1}
# High_impact_chronic_pain {0, 1}
for column in ['Chronic_Pain', 'High_impact_chronic_pain']:
    print(column, set(cleaned_data[column]), cleaned_data[column].value_counts().values)
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

### Auto ML ###
# tpot
# pycaret
# h2o
# auto-sklearn
# autokeras
# autogluon
# Hyperopt-Sklearn
# Auto-ViML
# MLBox

# https://epistasislab.github.io/tpot/api/
print("Tpot")
model = TPOTClassifier(scoring='accuracy', verbosity=2, random_state=1, n_jobs=-1) # generations=5, population_size=50,
model.fit(X_train, y_train)
model.export('tpot_best_model.py')
# Best pipeline: XGBClassifier(CombineDFs(input_matrix, input_matrix), learning_rate=0.1, max_depth=3, min_child_weight=18, n_estimators=100, n_jobs=1, subsample=0.15000000000000002, verbosity=0)
exit(1)

# https://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html
x = X_train.columns
y = y_train.columns
# For binary classification, response should be a factor
y_train[y] = y_train[y].asfactor()
# Run AutoML for 20 base models
aml = H2OAutoML(max_models=20, seed=1)
aml.train(x=x, y=y, training_frame=train)
# View the AutoML Leaderboard
lb = aml.leaderboard
lb.head(rows=lb.nrows)
preds = aml.predict(test)
exit(1)