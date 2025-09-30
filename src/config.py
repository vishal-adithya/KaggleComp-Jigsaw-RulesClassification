import warnings
warnings.filterwarnings("ignore")

import nltk # type: ignore
from nltk.corpus import stopwords
nltk.download("stopwords")
STOPWORDS_EN = set(stopwords.words("english"))


import os
TRAIN_DF_FILEPATH = os.path.join("../Data","train.csv")
TEST_DF_FILEPATH = os.path.join("../Data","test.csv")
SAMPLE_SUB_DF_FILEPATH = os.path.join("../Data","sample_submission.csv")