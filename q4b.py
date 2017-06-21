import numpy as np
import operator
import matplotlib as plt
import pandas as pd
import re
import csv
from pandas.core.frame import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
from numpy import linalg as LA
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr
import time

class Q4B:
    
    def __init__(self):
        self.simlex_path = "./SimLex-999.txt"
        self.counter_fitted_path   = "./counter-fitted-vectors.txt"
        self.glove_50d_path        = "./glove/glove.6B.50d.txt"

        #self.vocab_set = self.__get_vocab_set()
        self.simlex = self.build_simlex(self.simlex_path)
        self.glove_df = self.build_norm_table(self.glove_50d_path)
        self.cfv_df = self.build_norm_table(self.counter_fitted_path)

        simlex_list_gl = []
        simlex_list_cfv = []
        glove_list = []
        cfv_list = []

        i= 0
        for x in self.simlex.iterrows():
            w1    = x[1]['word1']
            w2    = x[1]['word2']
            v_sl  = x[1]['SimLex999']
            
            v_gl  = self.get_v_diff(self.glove_df, w1, w2)
            v_cfv  = self.get_v_diff(self.cfv_df, w1, w2)
            
            if v_gl is not None:
                simlex_list_gl.append(v_sl)
                
                #Calculate cosine similarity and return top 5 and lowest for each y
                glove_sim = cosine_similarity(self.glove_df.ix[w1].values.reshape(1,-1), self.glove_df.ix[w2].values.reshape(1,-1))[0][0]                
                glove_list.append(glove_sim)
                i += 1
                
            if v_cfv is not None:
                simlex_list_cfv.append(v_sl)
                cfv_sim = cosine_similarity(self.cfv_df.ix[w1].values.reshape(1,-1), self.cfv_df.ix[w2].values.reshape(1,-1))[0][0]
                cfv_list.append(cfv_sim)
        
        print(spearmanr(simlex_list_gl, glove_list).correlation)
        print(spearmanr(simlex_list_cfv, cfv_list).correlation)
        
        
        
    def get_v_diff(self, df, w1, w2):
        try:
            v = df.ix[w1]-df.ix[w2]
            return v
            
        except:
            return None
            
            
            
    def build_simlex(self, path):
        
        df = pd.read_csv( path, 
                            sep="\t", 
                            #index_col = 0,
                            header = 0,
                            quoting=csv.QUOTE_NONE)
#     
        return df
# 
    def build_norm_table(self, path):
         
        df = pd.read_table(path, 
                            sep=" ", 
                            index_col = 0,
                            header = None,
                            quoting=csv.QUOTE_NONE)
        
        
        df.index = df.index.str.lower()
#         df_filter = df.index.isin(list(self.vocab_set))
#         df_filtered = df[df_filter]
         
        #df_filtered = pd.DataFrame(df.loc[df['Character'].isin(self.vocab_set)])
         
        #df[df.index.map(lambda x: x[0] in self.vocab_set)]
#         df_norm = DataFrame(data=normalize(df), index=df.index)
         
        return df
def main():
    
    q = Q4B()
    
    
if __name__ == '__main__':
    start_time = time.time()

    main()

    print("Calc time: %s s" % (time.time()-start_time)) 
    