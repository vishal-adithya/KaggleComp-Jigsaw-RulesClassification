import warnings
warnings.filterwarnings("ignore")

import os
import re
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from utils import Preprocessing
from config import *

TRAIN_DF_FILEPATH = os.path.join("..","Data","train.csv")
TEST_DF_FILEPATH = os.path.join("..","Data","test.csv")
SAMPLE_SUB_DF_FILEPATH = os.path.join("..","Data","sample_submission.csv")

def Preprocessing(s):
    s = str(s)
    s = s.lower()
    _s = ""
    for i in s.split():
        if i not in STOPWORDS_EN:
            _s+=i+" "
            
    s = re.sub(r"http\S+", " ", _s)
    s = re.sub(r"@\w+", " ", s)               
    s = re.sub(r"[^a-z0-9\s]", " ", s)            
    s = re.sub(r"\s+", " ", s).strip()
    return s

train_df = pd.read_csv("../Data/train.csv")
train_df["test"] = train_df["body"].map(Preprocessing)
train_df[["test","body"]]
