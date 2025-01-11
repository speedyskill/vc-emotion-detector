import pandas as pd
import numpy as np
import pickle 
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, roc_auc_score
import json

#reading model in binary mode
clf=pickle.load(open('models/model.pkl','rb'))

test_df=pd.read_csv('./data/processed/test_bow.csv')

X_test = test_df.iloc[:, :-1].values  # Features
y_test = test_df.iloc[:, -1].values   # Target labels

# Make predictions
y_pred = clf.predict(X_test)
y_pred_proba = clf.predict_proba(X_test)[:, 1]  # Probabilities for the positive class

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

metrics_dict={
    'accuracy_score':accuracy,
    'precision_score':precision,
    'recall_score':recall,
    'auc_score':auc
}

with open('reports/metrics.json','w') as file:
    json.dump(metrics_dict,file,indent=4)
