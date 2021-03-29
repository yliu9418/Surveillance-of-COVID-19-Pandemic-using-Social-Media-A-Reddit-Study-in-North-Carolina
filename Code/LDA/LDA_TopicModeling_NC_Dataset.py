#!/usr/bin/env python
# coding: utf-8

import re, string, unicodedata
import spacy
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import os
import numpy as np
import re
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rc
from wordcloud import WordCloud

get_ipython().run_line_magic('matplotlib', 'inline')

subred_list = ['Charlotte', 'raleigh','gso','NorthCarolina']

def getTopicModel(sName):
    terms_count = 15
    num_topics = 5
    font = r'C:\Windows\Fonts\msyh.ttc'
    df = pd.read_csv(os.getcwd() + r'/reddit_data/%s.csv'%sName, encoding = 'utf-8')
    if df.dropna().empty:
        return
    cv = CountVectorizer(max_df=0.90, min_df=3, stop_words='english')
    try:
        td = cv.fit_transform(df['comments'].astype('U').values)
    except:
        return
    LDA = LatentDirichletAllocation(n_components=num_topics, max_iter=15,learning_method='online',learning_offset=15,random_state=42)
    LDA.fit(td)
    cv_num = str(len(cv.get_feature_names()))
    terms = cv.get_feature_names()
    myfile = open(os.getcwd() + r'/TM_Results/Text/%s.txt'%sName,'w+', encoding = 'utf-8')
    myfile.writelines('Total # of Unique Vocabulary for %s.csv is %s\n\n'%(sName,cv_num))
    myfile.writelines(('Unique Vocabulary for %s.csv\n\n'%sName +', '.join(cv.get_feature_names())+'\n\n'))
    for index,topic in enumerate(LDA.components_):
        abs_topic = abs(topic)
        topic_terms = [[terms[i],topic[i]] for i in abs_topic.argsort()[:-terms_count-1:-1]]
        topic_terms_sorted = [[terms[i], topic[i]] for i in abs_topic.argsort()[:-terms_count - 1:-1]]
        topic_words = []
        myfile.writelines(f'THE TOP %s WORDS FOR TOPIC {index + 1}: '%num_topics)
        try:
            for i in range(terms_count):
                topic_words.append([topic_terms_sorted[i][0],str(topic_terms_sorted[i][1].round(3))])
        except:
            break
       
        res = [', '.join(ele) for ele in topic_words] 
        myfile.writelines(' | '.join(res))
        myfile.writelines('\n\n')
        dict_word_frequency = {}

        for i in range(terms_count):
            dict_word_frequency[topic_terms_sorted[i][0]] = topic_terms_sorted[i][1]    
        wc = WordCloud(background_color="white",mask=None, max_words=100,
                            max_font_size=60,min_font_size=10,prefer_horizontal=0.9,
                            contour_width=3,contour_color='black',colormap ='copper',collocations=False)
        
        wc.generate_from_frequencies(dict_word_frequency) 
        plt.imshow(wc, interpolation='bilinear')
        plt.title('r/%s Topic %s'%(sName,index+1), fontweight='bold', fontdict={'fontsize' : 20}, loc='center', pad=None)
        plt.axis("off")
        output_dir=os.getcwd() + r'/TM_Results/Images/%s'%sName
        
        try: 
            os.makedirs(output_dir, exist_ok = True) 
            #print("Directory '%s' created successfully" %output_dir) 
        except OSError as error: 
            print("Directory '%s' can not be created")
                      
        plt.savefig(output_dir+'/%s_Topic_%s.png'%(sName,index+1), bbox_inches='tight')
    myfile.close()
    
    topic_results = LDA.transform(td)
    df['Topic#'] = topic_results.argmax(axis=1) + 1
    df['Largest Probability'] = np.amax(topic_results,axis=1).round(2)
    df.sort_values('Date', inplace = True, ascending = False)
    df.to_csv(os.getcwd() + r'/TM_Results/CSV/%s.csv'%sName, index=False, encoding="utf-8")

for i in subred_list:
    getTopicModel(i)

