from sklearn.datasets import load_wine
X, y = load_wine(return_X_y=True)


import matplotlib.pyplot as plt
import numpy as np
X_slice =  X[:,:2]
X_plot = np.transpose(X_slice)

# Select the first two attributes for a 2D representation of the image.
plt.scatter(X_plot[0],X_plot[1],c=y)
plt.title("Plot of the data according to the class:")
plt.xlabel("Alcohol")
plt.ylabel("Malic acid")
plt.show()


from sklearn.model_selection import train_test_split
#split in train+validation and test sets -> 70:30
X_trainValidation,X_test, y_trainValidation, y_test = train_test_split(X_slice, y,test_size=0.30, random_state=3, stratify=y)

#split in train and validation sets in order to have eventually a percentage of  50:20:30 respectively for train,validation and test sets
X_train,X_Validation,y_train,y_Validation = train_test_split(X_trainValidation,y_trainValidation,test_size=0.2857,random_state=3,stratify=y_trainValidation)

#Standardization of data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

scaler.fit(X_train) #Compute the mean and std to be used for later scaling.

#Perform standardization by centering and scaling
X_train = scaler.transform(X_train) 
X_Validation = scaler.transform(X_Validation)

scaler.fit(X_trainValidation)
X_test = scaler.transform(X_test)
X_trainValidation=scaler.transform(X_trainValidation)



C = [0.001, 0.01, 0.1, 1, 10, 100,1000]

#from sklearn.svm import SVC
from sklearn.svm import SVC
from mlxtend.plotting import plot_decision_regions
import matplotlib.gridspec as gridspec
import itertools

accuracyModel = {} #dict of accuracy values linked to the clfs key:index value:accuracy

labels=[] #vector of labels for the plot

model=[]  # vector of the clfs (classifiers)

for i in C : 
    model.append(SVC(C=i,kernel='linear',max_iter=10**9)) #definition of the clfs
    labels.append(f"Decision regions for Linear-SVC with C={i}")

#Decision boundaries
gs = gridspec.GridSpec(3,3) #grid 3x3
fig = plt.figure(figsize=(20,30)) # window to visualize the content of the plots


i=0
for grd, clf, lab in zip(itertools.product([0,1,2],repeat=2),model,labels): # product[a,b] repeat=2 => a,a a,b b,a b,b  #Cartesian product of input iterables. Equivalent to nested for-loops.
    clf.fit(X_train, y_train)
    ax = plt.subplot(gs[grd[0], grd[1]])
    fig = plot_decision_regions(X=X_train, y=y_train, clf=clf, legend=0)
    
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles,['Wine 1','Wine 2','Wine 3'],framealpha=0.3, scatterpoints=1)

    plt.title(lab)
    plt.xlabel("Alcohol")
    plt.ylabel("Malic acid")
    i+=1
plt.show()
####

labels =[]

for i in C : 
    labels.append(f"3-Class classification with Linear-SVC(C={i})")



gs = gridspec.GridSpec(3,3) #grid 3x3
fig = plt.figure(figsize=(20,30)) # window to visualize the content of the plots


i=0
for grd, clf, lab in zip(itertools.product([0,1,2],repeat=2),model,labels): # product[a,b] repeat=2 => a,a a,b b,a b,b  #Cartesian product of input iterables. Equivalent to nested for-loops.
    clf.fit(X_train, y_train)
    ax = plt.subplot(gs[grd[0], grd[1]])
    fig = plot_decision_regions(X=X_Validation, y=y_Validation, clf=clf, legend=0)
    
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles,['Wine 1','Wine 2','Wine 3'],framealpha=0.3, scatterpoints=1)
    
    accuracyModel[i] = round(clf.score(X_Validation,y_Validation),3)
    print( f"The accuracy of the model with C equal to {C[i]} is: {accuracyModel[i]}")
    plt.title(lab)
    plt.xlabel("Alcohol")
    plt.ylabel("Malic acid")
    i+=1
plt.show()

#Plot a graph showing how the accuracy on the validation set varies when changing C
accuracylist = sorted(accuracyModel.items()) # sorted by key, return a list of tuples
valueC, accuracy = zip(*accuracylist) # unpack a list of pairs into two tuples objects (key and value)
plt.plot(C, accuracy,'bo--')  #plt.plot(log(valueC), accuracy) per scalarli meglio

plt.xscale("log")
plt.ylabel('Accuracy level')
plt.xlabel('C')
plt.show()

#pick the best K and evaluate the model performance on the test
bestC = 0
bestV = 0

for key,value in accuracylist:
    if(value>bestV):
        bestV = value
        bestC = key

model = SVC(C=bestC,kernel='linear',max_iter=10**9)
model.fit(X_trainValidation, y_trainValidation) 
fig = plt.figure(figsize=(20,30)) # window to visualize the content of the plots
fig = plot_decision_regions(X=X_test, y=y_test, clf=model, legend=2)
plt.title(f"Classification of the best Linear SVC(C = {C[bestC]}) on test set")
plt.xlabel("Alcohol")
plt.ylabel("Malic acid")
plt.show()

print(f'The SVM built with C = {C[bestC]} has an accuracy on the test set up to: {round(model.score(X_test,y_test),3)}')
