#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python

import sys,os
import pyspark
import numpy as np
import operator
if len(sys.argv) != 5:
    sys.stderr.write("Need training labels, training data, test data, and output file name\n")
    sys.exit(1)

sc2 = pyspark.SparkContext.getOrCreate()
sc2.setLogLevel("WARN")
    
#extract the labels of the training set
lfile = sys.argv[1]

# read train and test set
trainfile=sys.argv[2]
testfile = sys.argv[3]
#get 1s and 2s for trainfile
f = sc2.textFile(trainfile)
datardd1=f.map(lambda x: x.split(' ')).map(lambda y: ((y[0]), int(y[1]), int(y[2])))
datardd1.persist()
datardd2=datardd1.filter(lambda x:x[2]!=0)
datardd2.persist()
datarddtrain=datardd2.map(lambda x: (x[0], x[1])).groupByKey().mapValues(list)
datarddtrain.persist()
# datardd4 is our trainset
#now let's get the test set
ftest=sc2.textFile(testfile)
datardd11=ftest.map(lambda x: x.split(' ')).    map(lambda y: ((y[0]), int(y[1]), int(y[2])))
datardd11.persist()
datardd12=datardd11.filter(lambda x:x[2]!=0)
datardd12.persist()
datarddtest=datardd12.map(lambda x: (x[0], x[1])).groupByKey().mapValues(list)
datarddtest.persist()
#get cartesian product
cart=datarddtrain.cartesian(datarddtest)
cart.persist()
#define jaccard similarity
from math import*
 
def jaccard_similarity(x,y):
 
    intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
#    union_cardinality = len(set.union(*[set(x), set(y)]))
    return intersection_cardinality

# define function for mapping jaccard similarity for each row in cartesian product
def maprowjaccard(row):
    jc=jaccard_similarity(row[0][1],row[1][1])
    tr=row[0][0]
    ts=row[1][0]
    return ts,(tr,float(jc))
cartnew=cart.map(maprowjaccard)
cartnew.persist()
#get labels

flab = sc2.textFile(lfile)
datarddlab=flab.map(lambda x: x.split(' '))
cartnew2=cartnew.groupByKey().mapValues(list)
cartnew2.persist()
#define function to groupby and remove test instance repeating in each nested list's row element
import operator
def sortgroups(row):
    row[1].sort(key=operator.itemgetter(1),reverse=True)
    return row
cartnew3=cartnew2.map(sortgroups)
cartnew3.persist()
#get only the closest 1 element or k=1 ,chance for overfitting but fastest
cartnew4=cartnew3.map(lambda x:(x[0],x[1][0]))
#this will get the highest or closest neighbour and not include the score
cartnew5=cartnew4.map(lambda x: (x[0],(x[1][0]) ))
#let's get the prediction file
predictDICT=cartnew5.map(lambda x:(x[1],x[0])).join(datarddlab)
predictfinal1=predictDICT.map(lambda x:(x[1][0],x[1][1]))
testnames = sc2.textFile(testfile).map(lambda line: line.split()[0]).distinct().collect()
#get testnames
predictfinal11=predictfinal1.collectAsMap()
predictfinal111=dict((testnames,predictfinal11[testnames]) for testnames in testnames if testnames in predictfinal11)
# list = [(k, v) for k, v in dict.items()]

newpredict=[(k,v) for k,v in predictfinal111.items()]
out = open(sys.argv[4],'w')
counter=0
for name in testnames:
    
    out.write('%s %s\n' % (newpredict[counter][0],newpredict[counter][1]))
    counter=counter+1
print(counter)

