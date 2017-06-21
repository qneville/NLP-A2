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
from nltk.metrics import spearman
import time

class Analogy:
    
    def __init__(self):
        
        self.master_analogies_path = "./questions-words.txt"
        self.counter_fitted_path   = "./counter-fitted-vectors.txt"
        self.glove_50d_path        = "./glove/glove.6B.50d.txt"
        
        self.vocab_set = self.__get_vocab_set()
        
        #cv = CountVectorizer()
        
        self.questions = self.build_questions(self.master_analogies_path)
        #self.cfv_df = self.build_norm_table(self.counter_fitted_path)
        self.glove_df = self.build_norm_table(self.glove_50d_path)
        
        # Do for all Counter-Fitted Vectors
        
        top5match = 0;
        y_a = {}
        i = 0;
        for x in self.questions.iterrows():
            Xa = x[1]['Xa'].lower()
            Xb = x[1]['Xb'].lower()
            Xc = x[1]['Xc'].lower()
            
            # y will return null if any Xa, Xb, Xc not found
            #y = self.__find_y(Xa, Xb, Xc)
            y = self.__calc_y(Xa, Xb, Xc)#self.cfv_df.ix[Xa]-self.cfv_df.ix[Xb]+self.cfv_df.ix[Xc]
            
            if y is not None:
                y_a[i] = {-1: "garbage"}
                for other in self.glove_df.iterrows():
                    if other[0] not in [Xa, Xb, Xc]: 
                        Y = y
                        Y2D = Y.values.reshape(1,-1)
                        X = other[1]
                        X2D = X.values.reshape(1,-1)
                        
                        #Calculate cosine similarity and return top 5 and lowest for each y
                        cos_sim2s = cosine_similarity(Y2D, X2D)[0][0]
                        
                        y_a[i][cos_sim2s] = other[0] #cos_sim)
                    
            
                # Report top 5 word vectors and if xD occurred.
                #if (i < 505):
                top5match += self.__get_print_top_5(x, y_a[i])
                #else: 
                   # print("Yo, start walking through code.")
                #print("\t\t\t...{}...".format(i))
                i += 1
                
                #if i%100 == 0:
#                     print(i)
                    
            
            print("-----------------------------")
            print("% Top 5 occurrences: {:.3}%".format(top5match/i*100))
            
    # Return 1 if match found in top 5
    def __get_print_top_5(self, x, answers_list ):
        #print("Top 5: "+x[1]['Xa']+" => "+x[1]['Xb']+" : "+x[1]['Xc']+" => "+x[1]['Xd'])
        return_val = 0;
        rank_i = 0;
        for y in sorted(answers_list, reverse=True):
            if rank_i < 5:
                ans = answers_list[sorted(answers_list, reverse = True)[rank_i]]
                if(ans.lower() == x[1]['Xd'].lower()): return_val = 1
                #print('\t\t{}: {}'.format(rank_i+1, ans))
            else:
                break
            rank_i+=1;
        
        return return_val(spearman.ranks_from_sequence)
    
    def __calc_y(self, Xa, Xb, Xc):
        try:
            return self.glove_df.ix[Xb]-self.glove_df.ix[Xa]+self.glove_df.ix[Xc]
            
        except:
            return None
            
    
    
    def build_questions(self, path):
        
        sect_reg = re.compile(u'^: \S+$')
        df_lines = []
        with open(path) as f:
            for line in f.read().split('\n'):
                if not(sect_reg.search(line)):
                    df_lines.append(line.split(" "))
            
        df = DataFrame(data = df_lines,
                       columns = ('Xa', 'Xb', 'Xc', 'Xd'))
                
        return df
    
    def build_norm_table(self, path):
        
        df = pd.read_table(path, 
                            sep=" ", 
                            index_col = 0,
                            header = None,
                            quoting=csv.QUOTE_NONE)
        
        df.index = df.index.str.lower()
        df_filter = df.index.isin(list(self.vocab_set))
        df_filtered = df[df_filter]
        
        #df_filtered = pd.DataFrame(df.loc[df['Character'].isin(self.vocab_set)])
        
        #df[df.index.map(lambda x: x[0] in self.vocab_set)]
        df_norm = DataFrame(data=normalize(df_filtered), index=df_filtered.index)
        
        return df_norm
    
    def __get_vocab_set(self):
        # We need the vocabulary first - use the vectorizer to get it from questions-words.txt
        vocab_array = []
    
        # Section label regex
        sect_reg = re.compile(u'^: \S+$')
        
        with open(self.master_analogies_path) as f:
            for line in f.read().split('\n'):
                if not(sect_reg.search(line)):
                    words = line.split(" ")
                    for word in words:
                        vocab_array.append(word.lower().rstrip())
                    
        return set(vocab_array)
    
def main():
    
    
    a = Analogy()    

        #do something
    # Normalize Something
    
    # Use this to establish a vocabulary
    
    # Fit GLOVE
    
    # Fit CounterFitted
    
    
if __name__ == '__main__':
    start_time = time.time()

    main()

    print("Calc time: %s s" % (time.time()-start_time)) 
    
        