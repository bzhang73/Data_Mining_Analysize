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
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Activation, Dense, Dropout, Input
from keras import optimizers, regularizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score, roc_curve, auc
from sklearn import preprocessing
from sklearn.preprocessing import minmax_scale



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
df = normilizedData.copy()

kmeans_data = normilizedData.copy()
kmeans_data.drop(0,axis=1,inplace=True)

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

def draw_hist_pca_sum(myList,Title,Xlabel,Ylabel,Xmin,Xmax,Ymin,Ymax):
    myList[0]=myList[0]*100;
    for i in range(1,len(myList)):
        myList[i]=myList[i-1]+myList[i]*100
        #    plt.hist(myList,len(myList))
#    for i in range(0,len(myList)):
        plt.plot(i+1,myList[i],marker='*');
    plt.xlabel(Xlabel)
    plt.xlim(Xmin,Xmax)
    plt.ylabel(Ylabel)
    plt.ylim(Ymin,Ymax)
    plt.title(Title)
    plt.show()


data1 = kmeans_data.values.tolist()
#print(data1)
x_std=StandardScaler().fit_transform(data1)
pca=PCA(.90)
pca_data_set=pca.fit_transform(x_std)

draw_hist_pca(pca.explained_variance_ratio_,'My pca variance ratio','X','Y',0,101,0,20);

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

#exit()
#print(len(pca_data_set[0]))
#pca_kmeans_dataset = dataset.copy()
#for i in range(2,39):
#    pca_kmeans_dataset.drop(i,axis=1,inplace=True)
#
#pca_kmeans_res = KMeans(n_clusters=7).fit(pca_data_set)
#for i in range(0,527):
#    pca_kmeans_dataset[0][i]=i
#    pca_kmeans_dataset[1][i]=pca_kmeans_res.labels_[i]
#print(pca_kmeans_dataset)


data2 = kmeans_data.iloc[:,0:17]
X = kmeans_data
# SCALE EACH FEATURE INTO [0, 1] RANGE
scale_X = minmax_scale(X, axis = 0)
#Build the AutoEncoder

train, test_df = train_test_split(scale_X, test_size = 0.25, random_state= 30)
train_df, dev_df = train_test_split(data2, test_size = 0.25, random_state= 30)

# Choose size of our encoded representations
encoding_dim = 17

# Define input layer
input_data = Input(shape=(scale_X.shape[1],))
# Define encoding layer
encoded = Dense(encoding_dim, activation='elu')(input_data)
# Define decoding layer
decoded = Dense(scale_X.shape[1], activation='sigmoid')(encoded)
# Create the autoencoder model
autoencoder = Model(input_data, decoded)
#Compile the autoencoder model
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
#Fit to train set, validate with dev set and save to hist_auto for plotting purposes
hist_auto = autoencoder.fit(train, train, epochs=50, batch_size=256, shuffle=True, validation_data=(test_df, test_df))

# Create a encoder in order to make encodings
encoder = Model(input_data, encoded)

# Create a placeholder for an encoded input
encoded_input = Input(shape=(encoding_dim,))
# Retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# Create the decoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))

# Encode and decode our test set
encoded_x = encoder.predict(test_df)
#decoded_output = decoder.predict(encoded_x)

autoencoder_res = pd.DataFrame(encoded_x[:,0:17])

distortions = []
inertias = []
mapping1 = {}
mapping2 = {}
for k in range (1,39):
    #construct the kMeans
    estimator = KMeans(n_clusters=k)
    estimator.fit(autoencoder_res)
    distortions.append(sum(np.min(cdist(autoencoder_res,estimator.cluster_centers_,'euclidean'),axis=1))/autoencoder_res.shape[0])
    inertias.append(estimator.inertia_)
    mapping1[k] = sum(np.min(cdist(autoencoder_res, estimator.cluster_centers_,
                                   'euclidean'),axis=1)) / autoencoder_res.shape[0]
    mapping2[k] = estimator.inertia_


for key, val in mapping1.items():
    print(str(key) + ' : ' + str(val))

X = range(1,39)
plt.title('The AutoEncoder Elbow Method using Distortion')
plt.xlabel('Value of K')
plt.ylabel('Distortion')
plt.plot(X,distortions,'bx-')
plt.show()

for key, val in mapping2.items():
    print(str(key) + ' : ' + str(val))

X = range(1,39)
plt.title('The AutoEncoder Elbow Method using Intertia')
plt.xlabel('Value ofK')
plt.ylabel('Intertia')
plt.plot(X,distortions,'bx-')
plt.show()

#exit()

#for i in range(1,8):
#    for j in range(0,527):
#        value=kmeans_res[j][i-1]
#        value=round(value,2)
#        kmeans_dataset[i][j]=value
#print(kmeans_dataset)
#print(kmeans_res)
#print(len(kmeans_res))

#outputpath='kmeans_output.data'
#kmeans_dataset.to_csv(outputpath,sep=',',index=False,header=False)

#writer = pd.ExcelWriter('b.xlsx')
#kmeans_dataset.to_excel(writer, sheet_name='Data')
#writer.save()

np.random.seed(527)

#Import data
df = pd.read_excel('data.xls', header = 1)
df = df.rename(columns = {'default value': 'Default'})

#preprocessed
#Check for missing values
df.isnull().sum() #No missing values thus no imputations needed

#Drop unneeded variables
#df = df.drop(['ID'], axis = 1)

print('AutoEncoder ')

categorical_columns = ['ID']

df = pd.get_dummies(df, columns = categorical_columns)

#Scale variables to [0,1] range
columns_to_scale = ['Q-E','ZN-E','PH-E','DBO-E','DQO-E','SS-E','SSV-E','SED-E','COND-E9','SS-P','PH-P','SSV-P','SSV-P','SED-P','COND-P','PH-D','DBO-D','DQO-D','SS-D','SSV-D','SED-D','COND-D','PH-S','DBO-S','DQO-S','SS-S','SSV-S','SED-S','COND-S','RD-DBO-P','RD-SS-P','RD-SED-P','RD-DBO-S','RD-DQO-S','RD-DBO-G','RD-DQO-G','RD-SS-G','RD-SED-G']

df[columns_to_scale]=df[columns_to_scale].apply(lambda x: (x-x.min())/(x.max()-x.min()))

#Split in 75% train and 25% test set
train, test_df = train_test_split(df, test_size = 0.25, random_state= 30)
train_df, dev_df = train_test_split(train, test_size = 0.25, random_state= 30)


# Check distribution of labels in train and test set
train_df.Default.sum()/train_df.shape[0]
dev_df.Default.sum()/dev_df.shape[0]
test_df.Default.sum()/test_df.shape[0]

# Define the final train and test sets
train_y = train_df.Default
dev_y = dev_df.Default
test_y = test_df.Default

train_x = train_df.drop(['Default'], axis = 1)
dev_x = dev_df.drop(['Default'], axis = 1)
test_x = test_df.drop(['Default'], axis = 1)

train_x =np.array(train_x)
dev_x =np.array(dev_x)
test_x = np.array(test_x)

train_y = np.array(train_y)
dev_y = np.array(dev_y)
test_y = np.array(test_y)

#Build the AutoEncoder

# Choose size of our encoded representations
encoding_dim = 17

# Define input layer
input_data = Input(shape=(train_x.shape[1],))
# Define encoding layer
encoded = Dense(encoding_dim, activation='elu')(input_data)
# Define decoding layer
decoded = Dense(train_x.shape[1], activation='sigmoid')(encoded)
# Create the autoencoder model
autoencoder = Model(input_data, decoded)
#Compile the autoencoder model
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
#Fit to train set, validate with dev set and save to hist_auto for plotting purposes
hist_auto = autoencoder.fit(train_x, train_x, epochs=50, batch_size=256, shuffle=True, validation_data=(dev_x, dev_x))

# Summarize history for loss
plt.figure()
plt.plot(hist_auto.history['loss'])
plt.plot(hist_auto.history['val_loss'])
plt.title('Autoencoder model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

# Create a encoder in order to make encodings
encoder = Model(input_data, encoded)

# Create a placeholder for an encoded input
encoded_input = Input(shape=(encoding_dim,))
# Retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# Create the decoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))

# Encode and decode our test set
encoded_x = encoder.predict(test_x)
decoded_output = decoder.predict(encoded_x)

#Build new model using encoded data
#Encode data set from above using the encoder
encoded_train_x = encoder.predict(train_x)
encoded_test_x = encoder.predict(test_x)

model = Sequential()
model.add(Dense(17, input_dim=encoded_train_x.shape[1], kernel_initializer='normal',
                #kernel_regularizer=regularizers.l2(0.02),
                activation="relu"
                )
          )
model.add(Dropout(0.2))
model.add(Dense(1))
model.add(Activation("sigmoid"))
model.compile(loss="binary_crossentropy", optimizer='adam')

history = model.fit(encoded_train_x, train_y, validation_split=0.2, epochs=10, batch_size=64)

# Summarize history for loss
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Encoded model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

#Predictions and visuallizations
#Predict on test set
predictions_NN_prob = model.predict(encoded_test_x)
predictions_NN_prob = predictions_NN_prob[:,0]

predictions_NN_01 = np.where(predictions_NN_prob > 0.5, 1, 0)
#Turn probability to 0-1 binary output

#Print accuracy
acc_NN = accuracy_score(test_y, predictions_NN_01)
print('Overall accuracy of Neural Network model:', acc_NN)





