import warnings
warnings.filterwarnings("ignore")

import os
import re
import joblib
import numpy as np 
import pandas as pd
import xgboost as xgb
from scipy.sparse import hstack as sparse_hstack,csr_matrix
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.metrics import classification_report,confusion_matrix

TRAIN_DF_FILEPATH = os.path.join("..","Data","train.csv")
TEXT_COLS = ["body",
             "negative_example_2","positive_example_1",
             "positive_example_2","negative_example_1"]
NUM_COLS = ["no advertising spam referral links unsolicited advertising and promotional content are not allowed",
            "no legal advice do not offer or request legal advice","subreddit"]

TEST_DF_FILEPATH = os.path.join("..","Data","test.csv")
SAMPLE_SUB_DF_FILEPATH = os.path.join("..","Data","sample_submission.csv")

def NLP_Preprocessing(s):
    s = str(s)
    s = s.lower()        
    s = re.sub(r"http\S+", " ", s)
    s = re.sub(r"@\w+", " ", s)               
    s = re.sub(r"[^a-z0-9\s]", " ", s)            
    s = re.sub(r"\s+", " ", s).strip()
    return s

train_df = pd.read_csv(TRAIN_DF_FILEPATH)
train_df.set_index(train_df["row_id"],inplace=True)
train_df.drop(columns = ["row_id"],inplace = True)
train_df.head()
train_df["body"] = train_df["body"].map(NLP_Preprocessing)
train_df["rule"] = train_df["rule"].map(NLP_Preprocessing)
train_df["positive_example_1"] = train_df["positive_example_1"].map(NLP_Preprocessing)
train_df["negative_example_2"] = train_df["negative_example_2"].map(NLP_Preprocessing)
train_df["positive_example_2"] = train_df["positive_example_2"].map(NLP_Preprocessing)
train_df["negative_example_1"] = train_df["negative_example_1"].map(NLP_Preprocessing)

def Preprocessing(df):
    rule_dummies = pd.get_dummies(df["rule"],dtype="float")
    le = LabelEncoder()
    df["subreddit"] = le.fit_transform(df["subreddit"])
    df.drop(columns = ["rule"],inplace = True)
    new_df = pd.concat([rule_dummies,df],axis = 1)
    return new_df

preprocessed_train_df = Preprocessing(train_df)

text_df = preprocessed_train_df[TEXT_COLS]
num_df = preprocessed_train_df[NUM_COLS]

text_df["positive"] = text_df["positive_example_1"] + " " + text_df["positive_example_2"]
text_df["negative"] = text_df["negative_example_1"] + " " + text_df["negative_example_2"]
text_df.drop(columns = ["negative_example_2",
                        "positive_example_1",
                        "positive_example_2",
                        "negative_example_1"],inplace = True)

tfidf = TfidfVectorizer(max_features=20000,ngram_range=(1,2))
tfidf.fit(text_df["negative"] + " " + text_df["positive"] + " "+ text_df["body"])
feature_1 = tfidf.transform(text_df["negative"].astype("str"))
feature_2 = tfidf.transform(text_df["positive"].astype("str"))
feature_3 = tfidf.transform(text_df["body"].astype("str"))

scaler = StandardScaler()
feature_4 = scaler.fit_transform(num_df.values)
feature_4 = csr_matrix(feature_4)
X_stack = sparse_hstack([feature_1,feature_2,feature_3,feature_4]).tocsr()
y = preprocessed_train_df["rule_violation"]

X_train,X_val,y_train,y_val = train_test_split(X_stack,y,
                                             random_state=4,
                                             test_size=0.2)

print(X_train.shape)
print(y_train.shape)
print(X_val.shape)
print(y_val.shape)

cls = xgb.XGBClassifier(random_state = 4,booster = "gbtree")
param_grid = {
    "n_estimators": [100,300,500],
    "learning_rate":[0.01,0.1,0.2],
    "max_depth": [3,6,9],
    "min_child_weight":[1,3,5],
    "subsample": [0.7,0.9,1.0],
    "colsample_bytree": [0.7,0.9,1.0],
    "reg_alpha":[0,0.01,0.1,1,10,100],
    "reg_lambda":[0.5,0.7,1.0,1.3]
}

rsv = RandomizedSearchCV(cls,param_distributions=param_grid,
                         n_iter=5,
                         cv = 10,
                         scoring="accuracy",
                         n_jobs=-1,
                         verbose=4,
                         random_state=4)

rsv.fit(X_train,y_train)
best_est =rsv.best_estimator_ 







yhat = best_est.predict(X_val)
print(classification_report(y_val,yhat))
print(confusion_matrix(y_val,yhat))
best_est.save_model("jgs-re-tfidf-stdscale-xgb-seed04--03-10-2025.json")
