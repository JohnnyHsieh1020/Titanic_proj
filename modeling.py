"""
Created on Thu Jun  3 23:08:29 2021

@author: Johnny Hsieh
"""
import pandas as pd
import numpy as np

df = pd.read_csv('dataset/full_data_modeling.csv')
test = pd.read_csv('dataset/test.csv')

# Get dummies
df_dummies = pd.get_dummies(df[['Pclass', 'Embarked', 'train_test',
       'new_cabin', 'Gender', 'Age_range', 'Family_count', 'LogFare',
       'Fare_range', 'Cabin_count', 'Sex_Pclass']])

# Split to train test again
X_train = df_dummies[df_dummies.train_test == 1].drop(['train_test'], axis =1)
X_test = df_dummies[df_dummies.train_test == 0].drop(['train_test'], axis =1)

y_train = df[df.train_test==1].Survived

# Modeling
# Model: Random Forest Classifier, Decision Tree Classifier, KNN, Naive Bayes Classifier, SVM
# XGboost、GridsearchCV
# import models
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.model_selection import GridSearchCV


# Random Forest Classifier
rf = RandomForestClassifier(random_state = 2, n_estimators=250, min_samples_split=20, oob_score=True)
cv = cross_val_score(rf, X_train, y_train, cv=5)
print(cv)
print(cv.mean()) # 0.810369719414977

# Decision Tree Classifier
dt = tree.DecisionTreeClassifier(random_state = 1)
cv = cross_val_score(dt, X_train, y_train,cv=5)
print(cv)
print(cv.mean()) # 0.804758018956751

# KNN
knn = KNeighborsClassifier()
cv = cross_val_score(knn, X_train, y_train,cv=5)
print(cv)
print(cv.mean()) # 0.8204695248258111

# Naive Bayes Classifier
gnb = GaussianNB()
cv = cross_val_score(gnb, X_train, y_train, cv=5)
print(cv)
print(cv.mean()) # 0.7115749168288243

# SVM
svc = SVC(probability = True)
cv = cross_val_score(svc, X_train, y_train, cv=5)
print(cv)
print(cv.mean()) # 0.8092084614901764

# XGboost
gbm = xgb.XGBClassifier(n_estimators= 2000, max_depth= 4, min_child_weight= 2, gamma=0.9, 
 subsample=0.8, colsample_bytree=0.8, objective= 'binary:logistic', nthread= -1, scale_pos_weight=1)

cv = cross_val_score(gbm, X_train, y_train, cv=5)
print(cv)
print(cv.mean()) # 0.8182286108844391


# GridSearchCV
# Random Forest Classifier
rf = RandomForestClassifier(random_state = 1)
param_grid = { 
    'n_estimators': [10, 50, 100, 300],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4, 5, 6, 7, 8],
    'criterion' :['gini', 'entropy']
}

rf_gs = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5)

# Decision Tree Classifier
dt = tree.DecisionTreeClassifier(random_state = 1)
param_grid = { 
   'criterion':['gini','entropy'],
   'max_depth': [10, 12, 14, 16, 18, 20]
}

dt_gs = GridSearchCV(estimator=dt, param_grid=param_grid, cv=5)

# KNN
knn = KNeighborsClassifier()
param_grid = { 
    'n_neighbors': [4, 5, 6, 7, 8],
    'weights': ['uniform', 'distance']
}

knn_gs = GridSearchCV(estimator=knn, param_grid=param_grid, cv=5)

# Naive Bayes Classifier
gnb = GaussianNB()
param_grid = { 
    'var_smoothing': np.logspace(0,-9, num=100)
}

gnb_gs = GridSearchCV(estimator=gnb, param_grid=param_grid, cv=5)

# SVM
svc = SVC(random_state = 1)
param_grid = {
    'C': [0.1, 1, 10, 100, 1000], 
    'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
    'kernel': ['rbf']
}

svc_gs = GridSearchCV(estimator=svc, param_grid=param_grid,refit = True, verbose = 3)

# XGboost
gbm = xgb.XGBClassifier()
param_grid = {
    'n_estimators': [10, 50, 100, 300],
    'min_child_weight': [1, 5, 10],
    'gamma': [0.5, 1, 1.5, 2, 5],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'max_depth': [3, 4, 5]
}

gbm_gs = GridSearchCV(estimator=gbm, param_grid=param_grid, cv=5)

# Fit
rf.fit(X_train,y_train)
dt.fit(X_train,y_train)
knn.fit(X_train,y_train)
gnb.fit(X_train,y_train)
svc.fit(X_train,y_train)
gbm.fit(X_train, y_train)

rf_gs.fit(X_train,y_train)
dt_gs.fit(X_train,y_train)
knn_gs.fit(X_train,y_train)
gnb_gs.fit(X_train,y_train)
svc_gs.fit(X_train,y_train)
gbm_gs.fit(X_train,y_train)

# Check best score
rf_gs.best_score_ # 0.8182097796748478
dt_gs.best_score_ # 0.8103822735547046
knn_gs.best_score_ # 0.8204695248258111
gnb_gs.best_score_ # 0.7700269914004143
svc_gs.best_score_ # 0.8159312033142928
gbm_gs.best_score_ # 0.8350574351892537

# Predict
pred_rf = rf.predict(X_test).astype(int)
pred_dt = dt.predict(X_test).astype(int)
pred_knn = knn.predict(X_test).astype(int)
pred_gnb = gnb.predict(X_test).astype(int)
pred_svc = svc.predict(X_test).astype(int)
pred_gbm = gbm.predict(X_test).astype(int)

pred_rf_gs = rf_gs.best_estimator_.predict(X_test).astype(int)
pred_dt_gs = dt_gs.best_estimator_.predict(X_test).astype(int)
pred_knn_gs = knn_gs.best_estimator_.predict(X_test).astype(int)
pred_gnb_gs = gnb_gs.best_estimator_.predict(X_test).astype(int)
pred_svc_gs = svc_gs.best_estimator_.predict(X_test).astype(int)
pred_gbm_gs = gbm_gs.best_estimator_.predict(X_test).astype(int)


# Check Performance
from sklearn.metrics import mean_absolute_error
default_ans = pd.read_csv('dataset/ans.csv')

score_rf = mean_absolute_error(default_ans.Survived, pred_rf) # MAE = 0.2631578947368421
score_dt = mean_absolute_error(default_ans.Survived, pred_dt) # MAE = 0.2511961722488038
score_knn = mean_absolute_error(default_ans.Survived, pred_knn) # MAE = 0.2799043062200957
score_gnb = mean_absolute_error(default_ans.Survived, pred_gnb) # MAE =  0.3373205741626794
score_svc = mean_absolute_error(default_ans.Survived, pred_svc) # MAE = 0.21770334928229665
score_gbm = mean_absolute_error(default_ans.Survived, pred_gbm) # MAE = 0.23444976076555024

score_rf_gs = mean_absolute_error(default_ans.Survived, pred_rf_gs) # MAE = 0.22248803827751196
score_dt_gs = mean_absolute_error(default_ans.Survived, pred_dt_gs) # MAE = 0.24641148325358853
score_knn_gs = mean_absolute_error(default_ans.Survived, pred_knn_gs) # MAE = 0.2799043062200957
score_gnb_gs = mean_absolute_error(default_ans.Survived, pred_gnb_gs) # MAE = 0.2822966507177033
score_svc_gs = mean_absolute_error(default_ans.Survived, pred_svc_gs) # MAE = 0.23923444976076555 
score_gbm_gs = mean_absolute_error(default_ans.Survived, pred_gbm_gs) # MAE = 0.23205741626794257
results = {
    'rf': score_rf, 
    'dt': score_dt, 
    'knn': score_knn, 
    'gnb': score_gnb, 
    'svc': score_svc, 
    'gbm': score_gbm, 
    'rf_gs': score_rf_gs, 
    'dt_gs': score_dt_gs, 
    'knn_gs': score_knn_gs, 
    'gnb_gs': score_gnb_gs, 
    'svc_gs': score_svc_gs, 
    'gbm_gs': score_gbm_gs
}

best_model = min(results, key=results. get)
print('The best model: ', best_model)
print('Score: ', results[best_model])
