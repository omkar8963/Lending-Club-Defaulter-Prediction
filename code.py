# --------------
import pandas as pd
from sklearn.model_selection import train_test_split


# Code starts here

# Read csv
df = pd.read_csv(filepath_or_buffer=path,compression='zip',low_memory=False)

# Load the data

X = df.drop('loan_status',1)
y = df.loan_status
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state = 4, test_size = 0.25)


# --------------
# Code starts  here
col = df.isnull().sum()
print(col.head())
col_drop = col[col>0.25*len(df)].index.tolist()
print(col_drop)
for x in  X_train:
    if X_train[x].nunique() == 1:
        col_drop.append(x)
X_train.drop(col_drop, axis=1, inplace=True)
X_test.drop(col_drop,axis=1, inplace=True)
print(X_train.head())
print(X_test.head())

# Code ends here


# --------------
import numpy as np


# Code starts here

y_train = np.where((y_train == 'Fully Paid') |(y_train == 'Current'), 0, 1)
y_test = np.where((y_test == 'Fully Paid') |(y_test == 'Current'), 0, 1)

# Code ends here


# --------------
from sklearn.preprocessing import LabelEncoder


# categorical and numerical variables
cat = X_train.select_dtypes(include = 'O').columns.tolist()
num = X_train.select_dtypes(exclude = 'O').columns.tolist()

# Code starts here
# Filling missing values

# Train Data

for x in cat:
    mode = X_train[x].mode()[0]
    X_train[x].fillna(mode, inplace = True)

for x in num:
    mean = X_train[x].mean()
    X_train[x].fillna(mean,inplace = True)

# Test Data
for x in cat:
    mode = X_train[x].mode()[0]
    X_test[x].fillna(mode,inplace = True)

for x in num:
    mean = X_train[x].mean()
    X_test[x].fillna(mean,inplace = True)


# Label encoding

le = LabelEncoder()
for x in cat:
    
    X_train[x] = le.fit_transform(X_train[x])
    X_test[x] = le.fit_transform(X_test[x])


# Code ends here


# --------------
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score,precision_score,recall_score,roc_auc_score,confusion_matrix,classification_report
from sklearn import metrics
import matplotlib.pyplot as plt

# Code starts here
rf = RandomForestClassifier(random_state =42,max_depth=2, min_samples_leaf=5000)
rf.fit(X_train,y_train)
y_pred = rf.predict(X_test)
f1 = f1_score(y_pred,y_test)
print("F1 Score: ",f1)
precision = precision_score(y_test,y_pred)
print("Precision Score: ",precision)
recall = recall_score(y_test,y_pred)
print("Recall Score: ",recall)
roc_auc = roc_auc_score(y_test,y_pred)
print("ROC:",roc_auc)
print("Confusion Matrix: ",confusion_matrix(y_pred,y_test))
print("Classification report: ",classification_report(y_pred,y_test))
score = roc_auc_score(y_pred,y_test)
y_pred_proba = rf.predict_proba(X_test)[:,1]
fpr , tpr , thresholds= metrics.roc_curve(y_test,y_pred_proba)
auc = roc_auc_score(y_test,y_pred_proba)
print(auc)
plt.plot(fpr,tpr,label="Random Forest model, auc="+str(auc))
plt.show()

# Code ends here


# --------------
from xgboost import XGBClassifier

# Code starts here
xgb = XGBClassifier(learning_rate = 0.0001)
xgb.fit(X_train,y_train)
y_pred = xgb.predict(X_test)
f1 = f1_score(y_pred,y_test)
precision = precision_score(y_test,y_pred)
recall = recall_score(y_test,y_pred)
roc_auc = roc_auc_score(y_test,y_pred)
print("Confusion Matrix: ",confusion_matrix(y_pred,y_test))
print("Classification report: ",classification_report(y_pred,y_test))
score = roc_auc_score(y_test,y_pred)
y_pred_proba = xgb.predict_proba(X_test)[:,1]
fpr , tpr, thresholds = metrics.roc_curve(y_test,y_pred_proba)
auc = roc_auc_score(y_test,y_pred_proba)
plt.plot(fpr,tpr,label="XGBoost model, auc="+str(auc))

# Code ends here


