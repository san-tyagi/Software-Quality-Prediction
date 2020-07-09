#!/usr/bin/env python
# coding: utf-8

# In[7]:


import os
import pandas as pd

def Naive_Bayes(data_train, data_test, target_train, target_test):
    from sklearn.naive_bayes import GaussianNB
    from sklearn.metrics import accuracy_score
    gnb = GaussianNB()
    pred = gnb.fit(data_train, target_train).predict(data_test)
    return accuracy_score(target_test, pred)

def Support_Vector_Machine(data_train, data_test, target_train, target_test):
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score
    svc_model = SVC(kernel='rbf',gamma='auto')
    pred = svc_model.fit(data_train, target_train).predict(data_test)
    return accuracy_score(target_test, pred)

def K_Nearest_Neighbors(data_train, data_test, target_train, target_test):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score
    neigh = KNeighborsClassifier(n_neighbors=20)
    neigh.fit(data_train, target_train)
    pred = neigh.predict(data_test)
    return accuracy_score(target_test, pred)

def Random_Forest(data_train, data_test, target_train, target_test):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    clf = RandomForestClassifier(max_depth=200, random_state=0)
    clf.fit(data_train, target_train)
    pred = clf.predict(data_test)
    return accuracy_score(target_test, pred)

def AdaBoost(data_train, data_test, target_train, target_test):
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn import metrics
    abc = AdaBoostClassifier(n_estimators=100,learning_rate=0.001)
    model = abc.fit(data_train, target_train)
    pred = model.predict(data_test)
    return metrics.accuracy_score(target_test, pred)

def Majority_Voting(data_train, data_test, target_train, target_test):
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.ensemble import VotingClassifier
    from sklearn import metrics

    nb_clf = GaussianNB()
    rf_clf = RandomForestClassifier()
    svm_clf = SVC()
    knn_clf = KNeighborsClassifier()
    ab_clf = AdaBoostClassifier()
    
    voting_clf = VotingClassifier(estimators = [('nb', nb_clf), ('rf', rf_clf), ('svm',svm_clf),('knn', knn_clf), ('ab', ab_clf)],voting = 'hard')
    model = voting_clf.fit(data_train, target_train)
    pred = model.predict(data_test)
    return metrics.accuracy_score(target_test, pred)

def SKB(data_out):
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import chi2
    from sklearn.model_selection import train_test_split
    import numpy as np
    NB=0
    SVM=0
    KNN=0
    RF=0
    AB=0
    MV=0
    m=len(data_out.columns)
    for i in range(m//5):
        array=data_out.values
        X=array[:,0:m-1]
        Y=array[:,m-1]
        test=SelectKBest(score_func=chi2, k=m-1-i)
        fit=test.fit(X, Y)
        cols=test.get_support(indices=True)
        l=[]
        j=0
        for x in cols:
            while(j!=x):
                l.append(data_out.columns[j])
                j=j+1
            j=j+1
        while(j<m):
            l.append(data_out.columns[j])
            j=j+1
        data=data_out
        Y=data[data.columns[m-1]]
        data=data.drop(l,axis='columns')
        data_nor=(data-data.min())/(data.max()-data.min())
        data=(data_nor-data_nor.mean())/data_nor.std()
        X=data[data.columns]
        data_train, data_test, target_train, target_test = train_test_split(X,Y, test_size = 0.20, random_state = 10)
        acc=Naive_Bayes(data_train, data_test, target_train, target_test)
        if(acc>NB):
            NB=acc
        acc=Support_Vector_Machine(data_train, data_test, target_train, target_test)
        if(acc>SVM):
            SVM=acc
        acc=K_Nearest_Neighbors(data_train, data_test, target_train, target_test)
        if(acc>KNN):
            KNN=acc
        acc=Random_Forest(data_train, data_test, target_train, target_test)
        if(acc>RF):
            RF=acc
        acc=AdaBoost(data_train, data_test, target_train, target_test)
        if(acc>AB):
            AB=acc
        acc=Majority_Voting(data_train, data_test, target_train, target_test)
        if(acc>MV):
            MV=acc
    l=[NB,SVM,KNN,RF,AB,MV]
    return l

def PCA(data_out):
    import numpy as np  
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    m=len(data_out.columns)
    X=data_out.iloc[:, 0:m-1].values 
    y=data_out.iloc[:, m-1].values 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) 
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train) 
    X_test = sc.transform(X_test)
    NB=0
    SVM=0
    KNN=0
    RF=0
    AB=0
    MV=0
    for i in range(min(m,len(X_train))-1):
        pca = PCA(n_components=i+1)
        data_train = pca.fit_transform(X_train)
        data_test = pca.transform(X_test)
        acc=Naive_Bayes(data_train, data_test, y_train, y_test)
        if(acc>NB):
            NB=acc
        acc=Support_Vector_Machine(data_train, data_test, y_train, y_test)
        if(acc>SVM):
            SVM=acc
        acc=K_Nearest_Neighbors(data_train, data_test, y_train, y_test)
        if(acc>KNN):
            KNN=acc
        acc=Random_Forest(data_train, data_test, y_train, y_test)
        if(acc>RF):
            RF=acc
        acc=AdaBoost(data_train, data_test, y_train, y_test)
        if(acc>AB):
            AB=acc
        acc=Majority_Voting(data_train, data_test, y_train, y_test)
        if(acc>MV):
            MV=acc
    l=[NB,SVM,KNN,RF,AB,MV]
    return l

def VAR(data_out):
    from sklearn.model_selection import train_test_split
    import numpy as np
    NB=0
    SVM=0
    KNN=0
    RF=0
    AB=0
    MV=0
    m=len(data_out.columns)
    l=[]
    for i in range(m-1):
        l.append([data_out[data_out.columns[i]].var(),i])
    l.sort()
    for i in range(m//5):
        ll=[]
        for j in range(i):
            ll.append(data_out.columns[l[i][1]])
        ll.append(data_out.columns[m-1])
        data=data_out
        Y=data[data.columns[m-1]]
        data=data.drop(ll,axis='columns')
        data=(data-data.mean())/data.std()
        X=data[data.columns]
        data_train, data_test, target_train, target_test = train_test_split(X,Y, test_size = 0.20, random_state = 10)
        acc=Naive_Bayes(data_train, data_test, target_train, target_test)
        if(acc>NB):
            NB=acc
        acc=Support_Vector_Machine(data_train, data_test, target_train, target_test)
        if(acc>SVM):
            SVM=acc
        acc=K_Nearest_Neighbors(data_train, data_test, target_train, target_test)
        if(acc>KNN):
            KNN=acc
        acc=Random_Forest(data_train, data_test, target_train, target_test)
        if(acc>RF):
            RF=acc
        acc=AdaBoost(data_train, data_test, target_train, target_test)
        if(acc>AB):
            AB=acc
        acc=Majority_Voting(data_train, data_test, target_train, target_test)
        if(acc>MV):
            MV=acc
    l=[NB,SVM,KNN,RF,AB,MV]
    return l        

import warnings
warnings.filterwarnings("ignore")
os.getcwd()
os.chdir('G:\\Android')
data=pd.read_csv(r'2014_1.csv')
n=len(data)
m=len(data.columns)
for i in range(m):
    if(data[data.columns[i]].dtypes=="object"):
        c=0
        a=0
        for j in range(n):
            if(data.iloc[j,i]!='?'):
                c=c+1
                a=a+float(data.iloc[j,i])
        a=a/c
        for j in range(n):
            if(data.iloc[j,i]=='?'):
                data.iloc[j,i]=a   
        data[data.columns[i]]=data[data.columns[i]].astype(float)
l=[]
for i in range(m):
    if(data[data.columns[i]].var()<=1e-25):
        l.append(data.columns[i])
data=data.drop(l,axis='columns')
m=len(data.columns)
Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1
data_bool=(data < (Q1 - 1.5 * IQR)) |(data > (Q3 + 1.5 * IQR))
lskb=[0,0,0,0,0,0]
lpca=[0,0,0,0,0,0]
lvar=[0,0,0,0,0,0]
for k in range(m-1):
    l=[]
    for i in range(n) : 
        c=0
        for j in range(m-1):
            if(data_bool.iloc[i,j]==True):
                c=c+1
        if(c>k):
            l.append(i)
    if(len(l)>=(n//12)):
        continue
    data_out=data
    for i in l:
        data_out=data_out.drop(i)
    l=[]
    for i in range(m):
        if(data_out[data_out.columns[i]].var()<=1e-25):
            l.append(data_out.columns[i])
    data_out=data_out.drop(l,axis='columns')
    acc=[]
    acc=SKB(data_out)
    if(acc[0]>lskb[0]):
        lskb[0]=acc[0]
    if(acc[1]>lskb[1]):
        lskb[1]=acc[1]
    if(acc[2]>lskb[2]):
        lskb[2]=acc[2]
    if(acc[3]>lskb[3]):
        lskb[3]=acc[3]
    if(acc[4]>lskb[4]):
        lskb[4]=acc[4]
    if(acc[5]>lskb[5]):
        lskb[5]=acc[5]
    acc=[]
    acc=PCA(data_out)
    if(acc[0]>lpca[0]):
        lpca[0]=acc[0]
    if(acc[1]>lpca[1]):
        lpca[1]=acc[1]
    if(acc[2]>lpca[2]):
        lpca[2]=acc[2]
    if(acc[3]>lpca[3]):
        lpca[3]=acc[3]
    if(acc[4]>lpca[4]):
        lpca[4]=acc[4]
    if(acc[5]>lpca[5]):
        lpca[5]=acc[5]
    acc=[]
    acc=VAR(data_out)
    if(acc[0]>lvar[0]):
        lvar[0]=acc[0]
    if(acc[1]>lvar[1]):
        lvar[1]=acc[1]
    if(acc[2]>lvar[2]):
        lvar[2]=acc[2]
    if(acc[3]>lvar[3]):
        lvar[3]=acc[3]
    if(acc[4]>lvar[4]):
        lvar[4]=acc[4]
    if(acc[5]>lvar[5]):
        lvar[5]=acc[5]
print(lskb,lpca,lvar)


# 
