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
import sys

class Q4A:
    
    def __init__(self):
        
        # Paths for files used
        self.master_analogies_path = "./questions-words.txt" # ~19544 lines
        self.counter_fitted_path   = "./counter-fitted-vectors.txt"
        self.glove_50d_path        = "./glove/glove.6B.50d.txt"
        
        # Used for metrics
        self.cfv_metrics = { 'total_read' : 0, 'in_top_5' : 0, 'in_top_1' : 0 }
        self.glove_metrics = { 'total_read' : 0, 'in_top_5' : 0, 'in_top_1' : 0 }

        # Get list of vocab from questions for pre-filtering Glove and CFV         
        self.vocab_set = self.__get_vocab_set()
        
        # Build Tables Needed
        self.questions = self.build_questions(self.master_analogies_path)
        self.cfv_df = self.build_norm_table(self.counter_fitted_path)
        self.glove_df = self.build_norm_table(self.glove_50d_path)
        
        # Do for all GLOVE_50 first
        print("Processing GLOVE_50:")        
        self.get_metrics(self.glove_metrics, self.glove_df, 0)       
        
        # Print Info
        print("Results:")
        print("-----------------------------")
        print("% Number of valid Question Analogies: {}".format(self.glove_metrics['total_read']))
        print("% Number of top 5's: {}".format(self.glove_metrics['in_top_5']))
        print("% Number of top 1's: {}".format(self.glove_metrics['in_top_1']))
        print("\n")
        
        print("% Occurs in top 5: {:.3}%".format(self.glove_metrics['in_top_5']/self.glove_metrics['total_read']*100))
        print("% Top result: {:.3}%".format(self.glove_metrics['in_top_1']/self.glove_metrics['total_read']*100))
        print("-----------------------------\n\n")
        
        
        print("Processing Counter-Fitted Vectors:")
        self.get_metrics(self.cfv_metrics, self.cfv_df, 0)
        print("Results:")
        
        print("-----------------------------")
        print("% Number of valid Question Analogies: {}".format(self.cfv_metrics['total_read']))
        print("% Number of top 5's: {}".format(self.cfv_metrics['in_top_5']))
        print("% Number of top 1's: {}".format(self.cfv_metrics['in_top_1']))
        print("")
        print("% Occurs in top 5: {:.3}%".format(self.cfv_metrics['in_top_5']/self.cfv_metrics['total_read']*100))
        print("% Top result: {:.3}%".format(self.cfv_metrics['in_top_1']/self.cfv_metrics['total_read']*100))
        print("-----------------------------\n\n")
        
    
    # Get totals necessary for percentages
    def get_metrics(self, metrics, source, limit=0):
        
        self.update_progress(0);
        
        y_a = {}
        i=0
        
        for x in self.questions.iterrows():
            Xa = x[1]['Xa'].lower()
            Xb = x[1]['Xb'].lower()
            Xc = x[1]['Xc'].lower()
            
            # Calculate Y vector
            y = self.__calc_y(source, Xa, Xb, Xc)
            
            # None of these words are missing from the Glove or CFV entries... Keep going.
            if y is not None:
                
                # Initialize empty dict here. 
                y_a[i] = {}
                
                # Now loop through Glove and CFV
                for other in source.iterrows():
                    
                    # Exclude Xa,Xb,Xc from maths
                    if other[0] not in [Xa, Xb, Xc]: 
                        
                        #Calculate cosine similarity and return top 5 and lowest for each y
                        cos_sim2s = cosine_similarity(y.values.reshape(1,-1), other[1].values.reshape(1,-1))[0][0]
                        
                        y_a[i][cos_sim2s] = other[0] #cos_sim)
                    
            
                # Report top 5 word vectors and if xD occurred.
                self.__analyze_top_5(x, metrics, y_a[i])
                
                metrics['total_read'] += 1
                i+=1
                
                if i%100 == 0:
                    self.update_progress(i);
            
            
            # Testing limit for debug    
            if limit is not 0 and i == limit:
                break
            
            
        
    # Return 1 if match found in top 5
    def __analyze_top_5(self, x, metrics, answers_list ):
        #print("Top 5: "+x[1]['Xa']+" => "+x[1]['Xb']+" : "+x[1]['Xc']+" => "+x[1]['Xd'])
        return_val = 0;
        rank_i = 0;
        for y in sorted(answers_list, reverse=True):
            if rank_i < 5:
                ans = answers_list[sorted(answers_list, reverse = True)[rank_i]]
                if(ans.lower() == x[1]['Xd'].lower()): 
                    metrics['in_top_5'] += 1
                    if rank_i == 0: metrics['in_top_1'] += 1 
            
            else:
                break
            rank_i+=1;
    
    def __calc_y(self, source, Xa, Xb, Xc):
        try:
            return source.ix[Xb]-source.ix[Xa]+source.ix[Xc]
            
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
        
        # Build table from either Glove or CFV
        df = pd.read_table(path, 
                            sep=" ", 
                            index_col = 0,
                            header = None,
                            quoting=csv.QUOTE_NONE)
        
        df.index = df.index.str.lower()
        df_filter = df.index.isin(list(self.vocab_set))
        df_filtered = df[df_filter]
        
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

    # Adapted from StackOverflow question:
    # https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
    def update_progress(self, progress_i):
        print('Approx: {}/19000'.format(progress_i))

def main():
    
    # Run Everything    
    Q4A()
    
    
if __name__ == '__main__':
    start_time = time.time()
    print"Executing Q4A..." 
    main()

    print("Total Time: {:.3} seconds".format(time.time()-start_time))
    
        