import os
import re
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
tqdm.pandas()
import nlplot
import warnings
warnings.simplefilter('ignore')    
import torch
import transformers
from transformers import BertJapaneseTokenizer
import logging
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR) # tokenize時の警告を抑制
def load_dataset():
    """データセットの読み込み"""
    paths = []
    for dirpath, dirnames, filenames in os.walk('./text'):
        for file in filenames:
            if re.match(r'.+[0-9].txt',  file):
                #print("{0}".format(file))
                paths.append(os.path.join(dirpath, file))                
    data = {
        'path': [],
        'URL': [],
        'date': [],
        'title': [],
        'text': [],
    }
    for path in paths:
        with open(path, 'r') as f:
            url = f.readline().strip('¥n')
            date = f.readline().strip('¥n')
            title = f.readline().strip('¥n')
            text = f.read()
            data['path'].append(path)
            data['URL'].append(url)
            data['date'].append(date)
            data['title'].append(title)
            data['text'].append(text)
    return pd.DataFrame(data)
df = load_dataset()
df['text_feature'] = pd.NA
#df.to_csv('df.csv', index=False)
df.to_pickle('df_obj.pkl')
