#please install sklearn numpy and matplotlib before use this file
#please pyt studentInfo.csv studentAssessment.csv under this folder
#to use this file , open terminal type
#python classifier.py

from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support
import csv
import numpy as np
import time
import matplotlib.pyplot as plt

#read csv data from studentInfo.csv and store it in data.
with open('studentInfo.csv') as f:
    f_csv = csv.reader(f)
    headers = next(f_csv)
    rows = [row for row in f_csv]
data = np.array(rows)


#read csv data from studentAssessment.csv and store it in assdata.
with open('studentAssessment.csv') as f:
    f_csv = csv.reader(f)
    headers = next(f_csv)
    rows = [row for row in f_csv]
assdata = np.array(rows)

#go through assdata , and create unique student id list
studentid = list(set(assdata[:,1]))

#create ass dictionary for store student id and its list of assement scores
ass = {}

#create list with studentid as key in dictionary
for items in studentid:
    ass[items] = []
rowa,columna = assdata.shape

#for each row find id and score , and append score into dictionary
for i in range(rowa):
    line = assdata[i,:]
    id = line[1]
    score = line[4]
    if score == '':
        score = 50
    ass[id].append(int(score))

#calculate mean of score list
for items in studentid:
    ass[items] = np.mean(ass[items])

row,column = data.shape
#init z column for assement data
z = np.zeros((row,1),dtype='int64')
data = np.concatenate((z,data),axis=1)


#plant assement data into data as a whole list
for items in range(row):
    line = data[items,:]
    id = line[3]
    try:
        assdata = ass[id]
    except:
        assdata = 50
    data[items,0] = assdata

#seperate result list as y
y = data[:,-1]


#drop useless attribute and put the rest together
assd = data[:,0:1]
a = data[:,1:3]
b = data[:,4:10]
c = data[:,11]
c=np.reshape(c,(-1,1))
x = np.concatenate((a,b),axis = 1)
x = np.concatenate((x,c),axis = 1)
x = np.concatenate((x,c),axis = 1)
row,column = x.shape



# function string to int, because training model dont accept data as string
def strings_to_int(xlist,row,column):
    for i in range(column):
        if type(xlist[0][i]) != int:
            a = list(set(xlist[:,i]))

            for j in range(len(a)):
                sub = a[j]
                for x in range(row):
                    if xlist[x][i] == sub:
                        xlist[x][i] =j
    return xlist

x = strings_to_int(x,row,column)

#combine x and assement data
x= np.concatenate((assd,x),axis=1)


#seperate the whole test set as train data ,validation data and test data.
train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=0.4,random_state=0)
test_x,validation_x,test_y,validation_y = train_test_split(test_x,test_y,test_size=0.5,random_state=0)


#this is a pltshow function, for ploting a graph to find out how to tune parameter
def pltshow():
    list1 = []
    for i in range(20):
        rf = RandomForestClassifier(n_estimators = 40,max_depth=12,min_samples_split=6,oob_score=True)
        rf.fit(train_x,train_y)
        score = rf.score(validation_x,validation_y)
        print(score)
        list1.append(score)

    plt.figure(figsize = (20,8),dpi = 80)
    plt.plot(range(1,21),list1)
    plt.show()






#this is a graph view function, for view ploted decision tree function
def graphview():
    classname = ['fail',"pass","withdrawn","distinction"]
    feature_name = ["code_module","code_presentation","gender","region","highest_education","imd_band","age_band","num_of_prev_attempts","disability"]
    dot_data = tree.export_graphviz(clf,feature_names= feature_name,filled = True,rounded = True,class_names=classname)
    graph = graphviz.Source(dot_data)
    graph.view()

#score function output model s performance on each dataset
def score(clf,train_x,train_y,test_x,test_y,validation_x,validation_y):
    score = clf.score(train_x,train_y)
    print('train:',score)
    score = clf.score(validation_x,validation_y)
    print('validation',score)
    score = clf.score(test_x,test_y)
    print('test',score)

#a function output precision recall and f1 score
def p(f,x,y):
    y1 = f.predict(x)
    a,b,c,d = precision_recall_fscore_support(y,y1,average='macro')
    print('precison:'+str(a)+'recall: '+str(b)+'f1 : '+str(c))

#main function train desicion tree , and ouput time used, score and precision recall f1
start = time.time()
clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_x,train_y)
end = time.time()
print('time:'+str(end-start))
score(clf,train_x,train_y,test_x,test_y,validation_x,validation_y)
p(clf,test_x,test_y)


#main function train randomforest , and ouput time used, score and precision recall f1
start = time.time()
rf = rf = RandomForestClassifier()
rf.fit(train_x,train_y)
end = time.time()
print('time:'+str(end-start))
score(rf,train_x,train_y,test_x,test_y,validation_x,validation_y)
p(rf,test_x,test_y)



#main function train desicion tree with fine tuned parameter , and ouput score and precision recall f1
clf = tree.DecisionTreeClassifier(max_depth=7,splitter="best",min_samples_leaf=10,min_samples_split=2)
clf = clf.fit(train_x,train_y)
score(clf,train_x,train_y,test_x,test_y,validation_x,validation_y)
p(clf,test_x,test_y)


#main function train random forest with fine tuned parameter , and ouput  score and precision recall f1
rf = rf = RandomForestClassifier(n_estimators = 40,max_depth=12,min_samples_split=6,oob_score=True)
rf.fit(train_x,train_y)
score(rf,train_x,train_y,test_x,test_y,validation_x,validation_y)
p(rf,test_x,test_y)