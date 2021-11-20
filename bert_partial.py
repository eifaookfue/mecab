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
logger = logging.getLogger(__name__)
fmt = "%(asctime)s %(levelname)s %(name)s :%(message)s"
logging.basicConfig(level=logging.INFO, format=fmt)
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR) # tokenize時の警告を抑制
import sys


class BertExtractor:
    """文書特徴抽出用クラス"""
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu' #GPUが使用可能ならGPUを使用
        self.model_name = 'cl-tohoku/bert-base-japanese-whole-word-masking' #使用する学習済みモデル名
        self.tokenizer = BertJapaneseTokenizer.from_pretrained(self.model_name) #使用するBERTトークナイザー
        self.bert_model = transformers.BertModel.from_pretrained(self.model_name) #学習済みモデル呼び出し
        self.bert_model = self.bert_model.to(self.device)
        self.max_len = 128 #使用する入力文書の長さ。最大512まで
    def extract(self, sentence):
        """文書特徴ベクトルを抽出する"""
        # 文書のトークナイズ
        inp = self.tokenizer.encode(sentence)
        len_inp = len(inp)
        # 入力トークン数の調整
        if len_inp >= self.max_len:
            inputs = inp[:self.max_len]
        else:
            inputs = inp + [0] * (self.max_len - len_inp)
        # モデルへ文書を入力し特徴ベクトルを取り出す
        inputs_tensor = torch.tensor([inputs], dtype=torch.long).to(self.device)        
        #seq_out, _ = self.bert_model(inputs_tensor)
        bert_out = self.bert_model(inputs_tensor)
        seq_out, pooled_out = bert_out['last_hidden_state'], bert_out['pooler_output']
        if torch.cuda.is_available():    
            #return seq_out[0][0].cpu().detach().numpy() # 0番目は [CLS] token, 768 dim の文章特徴量
            #return inputs_tensor
            return seq_out[0][0].cpu().detach().numpy()
        else:
            #return seq_out[0][0].detach().numpy()
            #return inputs_tensor
            return seq_out[0][0].detach().numpy()

def cos_sim_matrix(matrix):
    """文書間のコサイン類似度を計算し、類似度行列を返す"""
    d = matrix @ matrix.T 
    norm = (matrix * matrix).sum(axis=1, keepdims=True) ** .5
    return d / norm / norm.T

def execute(x):
	logger.info(x[0:20])
	return bex.extract(x)	

#df = pd.read_csv('df.csv')
df = pd.read_pickle('df_obj.pkl')
bex = BertExtractor()
fromIdx=int(sys.argv[1])
toIdx=int(sys.argv[2])
df['text_feature'][fromIdx:toIdx] = df['text'][fromIdx:toIdx].progress_apply(execute)
#df.to_csv('df.csv', index=False)
df.to_pickle('df_obj.pkl')
