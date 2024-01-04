import time

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
import h2o
from h2o.automl import H2OAutoML
import autokeras as ak
from autokeras import StructuredDataClassifier
import shap
# import shapley
from xgboost import XGBClassifier
# from lightgbm import LGBMClassifier


print("\ncleaned_data")
cleaned_data = pd.read_csv('Cleaned_data_2019.csv')
for column in cleaned_data.columns:
    print(column, set(cleaned_data[column]))
print('cleaned_data: ',cleaned_data.shape)
# Chronic_Pain {0, 1}
# High_impact_chronic_pain {0, 1}
outcomes = ['Chronic_Pain', 'High_impact_chronic_pain']
for column in outcomes:
    print(column, set(cleaned_data[column]), cleaned_data[column].value_counts().values)
outcome = ['High_impact_chronic_pain'] # 'Chronic_Pain', 'High_impact_chronic_pain'
drop_col = [x for x in outcomes if x not in outcome]
print("Outcome:",outcome," \nDropped_col:",drop_col)
cleaned_data.drop(drop_col, axis=1, inplace=True) # 'High_impact_chronic_pain'

# Modeling
print("\nModeling")
X = cleaned_data.drop(outcome, axis=1)  # Features
Y = cleaned_data[outcome]  # Target
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

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

print("XGboost") ############################### <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Chosen model
clf = XGBClassifier()
clf.fit(X, Y)
y_pred_clf = clf.predict(X)
accuracy = accuracy_score(Y, y_pred_clf)
print("Accuracy (XGBoost Classifier):", accuracy)

def custom_predict(X):
    return clf.predict(X)
kmeans_k =100 # 100
rows_devideby_to_use = 1 # 1
explainer = shap.KernelExplainer(custom_predict, shap.kmeans(X.values, kmeans_k))
number_of_rows = X.values.shape[0]
random_indices = np.random.choice(number_of_rows, size=number_of_rows//rows_devideby_to_use, replace=False)
random_rows = X.iloc[random_indices] #.values
print("explainer.shap_values")
shap_values = explainer.shap_values(random_rows)

print('training-ish size:', len(random_rows.values), len(random_rows.values[0]))
print('\nD1 Classes:', len(shap_values), '\nD2 samples:', len(shap_values[0]))#, '\nD3 Columns/features:', len(shap_values[0][0])) # , '\nvalue:', shap_values[0][0][0]
print('type: ',type(shap_values))
print('type [0]: ', type(shap_values[0]))

print("write shap_values")
for i in range(len(shap_values)):
    np.savetxt("./shap2/shap_"+str(i)+".csv", shap_values[i])
np.savetxt("./shap2/shape.csv",np.array([len(shap_values)]))

column_names = X.columns.values
print(column_names)
pd.DataFrame(column_names, columns=['Column Names']).to_csv('./shap2/columns.csv', index=False)

exit()

exit (-1) ############################### <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Old runs

# https://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html
print("h2o")  ############################### <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
h2o.init()
aml = H2OAutoML(max_models=3, seed=1) # before 20 eresult below
x=X.columns.tolist()
y=Y.columns.tolist()[0]
cleaned_data_h2o= h2o.H2OFrame(cleaned_data)
cleaned_data_h2o[y] = cleaned_data_h2o[y].asfactor()
print(len(x),x,"\n",y)

# aml.train(x=x, y=y, training_frame=cleaned_data_h2o)
# lb = aml.leaderboard
# lb.head(rows=lb.nrows)
# print("H2o: ",lb.head(rows=lb.nrows))
# best_model = h2o.get_model(lb[0, 'model_id'])
# model_path = best_model.save_mojo("./h2o_best_model_mojo")
# print(model_path) # /Users/macpro/PycharmProjects/NHIS/h2o_best_model_mojo/StackedEnsemble_BestOfFamily_1_AutoML_1_20240103_175628.zip
model_path = "/Users/macpro/PycharmProjects/NHIS/h2o_best_model_mojo/StackedEnsemble_BestOfFamily_1_AutoML_1_20240103_175628.zip"
best_model = h2o.import_mojo(model_path)

data_for_shap =cleaned_data_h2o.as_data_frame()
X_shap = data_for_shap.values # .drop(y, axis=1)
def predict_func(X):
    return best_model.predict(h2o.H2OFrame(X)).as_data_frame().values.flatten()
# 82 %
# model_id                                                      auc    logloss     aucpr    mean_per_class_error      rmse       mse
# StackedEnsemble_AllModels_1_AutoML_5_20240103_165139     0.823065   0.405868  0.622559                0.264175  0.358471  0.128501
# StackedEnsemble_BestOfFamily_1_AutoML_5_20240103_165139  0.822348   0.406775  0.620386                0.262379  0.358969  0.128859
# GBM_grid_1_AutoML_5_20240103_165139_model_2              0.821338   0.408253  0.618297                0.261707  0.359459  0.129211
# GBM_5_AutoML_5_20240103_165139                           0.8203     0.408981  0.616893                0.265317  0.359965  0.129575
# GBM_1_AutoML_5_20240103_165139                           0.820063   0.408395  0.61845                 0.263373  0.359695  0.129381
# GBM_2_AutoML_5_20240103_165139                           0.819374   0.410107  0.613996                0.266462  0.360541  0.12999
# GLM_1_AutoML_5_20240103_165139                           0.817865   0.41117   0.61155                 0.262916  0.361108  0.130399
# GBM_3_AutoML_5_20240103_165139                           0.81701    0.412856  0.606417                0.268459  0.362031  0.131066
# XGBoost_3_AutoML_5_20240103_165139                       0.816137   0.413028  0.608252                0.265806  0.362068  0.131093
# XGBoost_grid_1_AutoML_5_20240103_165139_model_3          0.814341   0.414698  0.606744                0.269712  0.362479  0.131391
# GBM_4_AutoML_5_20240103_165139                           0.813646   0.415417  0.604124                0.269013  0.363234  0.131939
# DRF_1_AutoML_5_20240103_165139                           0.808394   0.420776  0.596667                0.272009  0.36478   0.133065
# DeepLearning_1_AutoML_5_20240103_165139                  0.80735    0.421193  0.59573                 0.276512  0.36539   0.13351
# XGBoost_grid_1_AutoML_5_20240103_165139_model_2          0.805408   0.426518  0.593782                0.273987  0.367283  0.134897
# GBM_grid_1_AutoML_5_20240103_165139_model_1              0.804997   0.423626  0.584946                0.275394  0.367026  0.134708
# DeepLearning_grid_1_AutoML_5_20240103_165139_model_1     0.804793   0.435337  0.598059                0.272375  0.368002  0.135426
# DeepLearning_grid_3_AutoML_5_20240103_165139_model_1     0.803743   0.428389  0.596082                0.278294  0.366906  0.13462
# XRT_1_AutoML_5_20240103_165139                           0.80327    0.436694  0.585898                0.277114  0.371433  0.137962
# XGBoost_grid_1_AutoML_5_20240103_165139_model_1          0.801576   0.431547  0.581897                0.280946  0.370062  0.136946
# DeepLearning_grid_2_AutoML_5_20240103_165139_model_1     0.801469   0.430958  0.585982                0.273034  0.367842  0.135308
# XGBoost_2_AutoML_5_20240103_165139                       0.796801   0.439078  0.576644                0.277189  0.37254   0.138786
# XGBoost_1_AutoML_5_20240103_165139                       0.796613   0.438543  0.578245                0.283792  0.372661  0.138876
exit(1)

# XGboost
print("XGboost") ############################### <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
clf = XGBClassifier()
clf.fit(x_train, y_train)
y_pred_clf = clf.predict(x_test)
accuracy = accuracy_score(y_test, y_pred_clf)
print("Accuracy (XGBoost Classifier):", accuracy)
# clf2 = LGBMClassifier()
# clf2.fit(X_train, y_train)
# y_pred_clf2 = clf2.predict(x_test)
# accuracy2 = accuracy_score(y_test, y_pred_clf2)
# print("Accuracy (light XGBoost Classifier):", accuracy)
# 80%
# Accuracy (XGBoost Classifier): 0.8035485933503836
exit(-1)

# auto Keras
print("auto keras")  ############################### <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
search = StructuredDataClassifier(max_trials=2)
search.fit(x=x_train, y=y_train, verbose=0)
loss, acc = search.evaluate(x_test, y_test, verbose=0)
print('Accuracy auto keras: %.3f' % acc)
# %
# out
exit(-1)

# https://epistasislab.github.io/tpot/api/
print("Tpot")  ############################### <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
model = TPOTClassifier(scoring='accuracy', verbosity=2, random_state=1, n_jobs=-1) # generations=5, population_size=50,
model.fit(x_train, y_train)
model.export('tpot_best_model.py')
# 79%
# Best pipeline: XGBClassifier(CombineDFs(input_matrix, input_matrix), learning_rate=0.1, max_depth=3, min_child_weight=18, n_estimators=100, n_jobs=1, subsample=0.15000000000000002, verbosity=0)
# Best pipeline: XGBClassifier(CombineDFs(input_matrix, CombineDFs(CombineDFs(CombineDFs(GaussianNB(input_matrix), input_matrix), input_matrix), MLPClassifier(input_matrix, alpha=0.001, learning_rate_init=0.1))), learning_rate=0.1, max_depth=3, min_child_weight=1, n_estimators=100, n_jobs=1, subsample=0.5, verbosity=0)
exit(1)

# Random Forest
print("random forest") ############################### <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
clf = RandomForestClassifier(n_estimators=1000,random_state=42)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
print(f"Accuracy Random forest: {accuracy}")
print("Classification Report:")
print(report)
# 80%
# out
exit(-1)