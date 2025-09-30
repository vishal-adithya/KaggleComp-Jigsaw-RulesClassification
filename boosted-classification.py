import warnings
warnings.filterwarnings("ignore")

import nltk # type: ignore
nltk.download("stopwords")

import os
import re
from nltk.corpus import stopwords # type: ignore
import numpy as np # type:ignore
import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore

TRAIN_DF_FILEPATH = os.path.join("Data","train.csv")
STOPWORDS_ = set(stopwords.words("english"))


train_df = pd.read_csv(TRAIN_DF_FILEPATH)
train_df.set_index("row_id",inplace=True)

def Preprocessing(s):
    s = str(s)
    s = s.lower()
    _s = ""
    for i in s.split():
        if i not in STOPWORDS_:
            _s+=i+" "
            
    s = re.sub(r"http\S+", " ", _s)
    s = re.sub(r"@\w+", " ", s)               
    s = re.sub(r"[^a-z0-9\s]", " ", s)            
    s = re.sub(r"\s+", " ", s).strip()
    return s

train_df["test"] = train_df["body"].map(Preprocessing)
train_df[["test","body"]]