import numpy as np
import pandas as pd
import keras

from collections import Counter
from sklearn import preprocessing
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from keras.layers import Dense, Activation, Input
from keras.models import Model
from keras import regularizers


dataset = pd.read_csv('water-treatment.data',header=None)
print(dataset)
#print(dataset[0][0])

statics = pd.read_csv('statistics.data',header=None)
#statics = pd.DataFrame(ss)

#print(statics)
#print(statics[1][1])
fillData = dataset.copy()
normilizedData = dataset.copy()
#exit()

pd.set_option('mode.chained_assignment', None)

for i in range(1,39):
    min = float(statics[2][i-1])
    max = float(statics[3][i-1])
#    print(min   )
    for j in range(0,527):
        value = dataset[i][j]
        if value == '?':
#            print(i,' ',j,' ',dataset[i][j])
            mean = float(statics[4][i-1])
            fillData[i][j]=mean
#            print(dataset[i][j])
#            exit()
        value = float(fillData[i][j])
        normilizedData[i][j] = (value -min)/(max-min)

#            print(normilizedData[i][j])

#print(fillData)
#print()
print(normilizedData)

#sum of the squared errors
kmeans_data = normilizedData.copy()
kmeans_data.drop(0,axis=1,inplace=True)
#print(kmeans_data)
#exit()
distortions = []
inertias = []
mapping1 = {}
mapping2 = {}
for k in range (1,39):
    #construct the kMeans
    estimator = KMeans(n_clusters=k)
    estimator.fit(kmeans_data)
    distortions.append(sum(np.min(cdist(kmeans_data,estimator.cluster_centers_,'euclidean'),axis=1))/kmeans_data.shape[0])
    inertias.append(estimator.inertia_)
    mapping1[k] = sum(np.min(cdist(kmeans_data, estimator.cluster_centers_,
                                   'euclidean'),axis=1)) / kmeans_data.shape[0]
    mapping2[k] = estimator.inertia_


for key, val in mapping1.items():
    print(str(key) + ' : ' + str(val))

X = range(1,39)
plt.title('The Elbow Method using Distortion')
plt.xlabel('Value of K')
plt.ylabel('Distortion')
plt.plot(X,distortions,'bx-')
plt.show()

for key, val in mapping2.items():
    print(str(key) + ' : ' + str(val))

X = range(1,39)
plt.title('The Elbow Method using Intertia')
plt.xlabel('Value ofK')
plt.ylabel('Intertia')
plt.plot(X,distortions,'bx-')
plt.show()

kmeans_res = KMeans(n_clusters=7).fit(kmeans_data)
#print(kmeans_res.labels_[1])
#print(kmeans_res)
print(len(kmeans_res.labels_))

#print(len(kmeans_res[0]))
kmeans_dataset = dataset.copy()
for i in range(2,39):
    kmeans_dataset.drop(i,axis=1,inplace=True)
for i in range(0,527):
    kmeans_dataset[0][i]=i
    kmeans_dataset[1][i]=kmeans_res.labels_[i]
print(kmeans_dataset)


#for i in range(1,8):
#    for j in range(0,527):
#        value=kmeans_res[j][i-1]
#        value=round(value,2)
#        kmeans_dataset[i][j]=value
#print(kmeans_dataset)
#print(kmeans_res)
#print(len(kmeans_res))

outputpath='kmeans_output.data'
kmeans_dataset.to_csv(outputpath,sep=',',index=False,header=False)





data1 = kmeans_data.values.tolist()
#print(data1)
x_std=StandardScaler().fit_transform(data1)
pca=PCA(.90)
pca_data_set=pca.fit_transform(x_std)
#print(len(pca_data_set[0]))
pca_kmeans_dataset = dataset.copy()
for i in range(2,39):
    pca_kmeans_dataset.drop(i,axis=1,inplace=True)

pca_kmeans_res = KMeans(n_clusters=7).fit(pca_data_set)
for i in range(0,527):
    pca_kmeans_dataset[0][i]=i
    pca_kmeans_dataset[1][i]=pca_kmeans_res.labels_[i]
print(pca_kmeans_dataset)

outputpath='pca_kmeans_output.data'
kmeans_dataset.to_csv(outputpath,sep=',',index=False,header=False)


def draw_hist_pca(myList,Title,Xlabel,Ylabel,Xmin,Xmax,Ymin,Ymax):
    plt.figure()
    myList[0]=myList[0]*100;
    #    for i in range(1,len(myList)):
    #        myList[i]=myList[i-1]+myList[i]*100
    #    #    plt.hist(myList,len(myList))
    for i in range(0,len(myList)):
        plt.plot(i+1,myList[i],marker='*');
    plt.xlabel(Xlabel)
    plt.xlim(Xmin,Xmax)
    plt.ylabel(Ylabel)
    plt.ylim(Ymin,Ymax)
    plt.title(Title)

plt.show()

#draw_hist_pca(pca.explained_variance_ratio_,'My pca variance ratio','X','Y',0,101,0,20);

distortions = []
inertias = []
mapping1 = {}
mapping2 = {}
for k in range (1,39):
    #construct the kMeans
    estimator = KMeans(n_clusters=k)
    estimator.fit(pca_data_set)
    distortions.append(sum(np.min(cdist(pca_data_set,estimator.cluster_centers_,'euclidean'),axis=1))/pca_data_set.shape[0])
    inertias.append(estimator.inertia_)
    mapping1[k] = sum(np.min(cdist(pca_data_set, estimator.cluster_centers_,
                                   'euclidean'),axis=1)) / pca_data_set.shape[0]
    mapping2[k] = estimator.inertia_


for key, val in mapping1.items():
    print(str(key) + ' : ' + str(val))

X = range(1,39)
plt.title('The PCA Elbow Method using Distortion')
plt.xlabel('Value of K')
plt.ylabel('Distortion')
plt.plot(X,distortions,'bx-')
plt.show()

for key, val in mapping2.items():
    print(str(key) + ' : ' + str(val))

X = range(1,39)
plt.title('The PCA Elbow Method using Intertia')
plt.xlabel('Value ofK')
plt.ylabel('Intertia')
plt.plot(X,distortions,'bx-')
plt.show()
