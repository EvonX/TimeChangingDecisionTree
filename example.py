
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import collections
from collections import deque
from pandas import Timestamp
from tcdt import TCDT


# In[2]:


#load sample data
fname='sample.csv'
sampleData = pd.read_csv(fname, encoding='utf-8')
sampleData["b'timestamp'"] = pd.to_datetime(sampleData["b'timestamp'"])
sampleData


# In[3]:


# features for training, 
# ["b'prefix0'", "b'prefix1'", "b'prefix2'", "b'request_url_path'"] indicate the hirerarchy of the url features
trainFeatures = [["b'remote_host'"], 
                 ["b'remote_user'"], 
                 ["b'request_method'"],
                 ["b'prefix0'", "b'prefix1'", 
                  "b'prefix2'", "b'request_url_path'"]]
labelCol = "b'status'"
timeCol = "b'timestamp'"

#train a decision tree, 
trainData = sampleData[:10]
tcdt = TCDT().fit(trainData, trainFeatures, labelCol, timeCol)

#print out the tree
print(tcdt.export_text())


# In[4]:


# make a prediction, here the testData have a new change so the predictions are wrong
testData = sampleData[10:].to_numpy()
for row in testData:
    print(row)
    print('predict: ', tcdt.predict(row))


# In[5]:


# update the tcdt with the testData and the predictions become correct now
tcdt.update(testData[0])
for row in testData:
    print(row)
    print('predict: ', tcdt.predict(row))


# In[6]:


# print all tree leaves
for l in tcdt.get_allleaves():
    print(l.ipath)
    print(l.timeSeries)

