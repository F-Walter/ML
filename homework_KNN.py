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
X_trainValidation,X_test, y_trainValidation, y_test = train_test_split(X_slice, y, test_size=0.30, random_state=1, stratify=y)

#split in train and validation sets in order to have eventually a percentage of  50:20:30 respectively for train,validation and test sets
X_train,X_Validation,y_train,y_Validation = train_test_split(X_trainValidation,y_trainValidation, test_size=0.2857,random_state=1,stratify=y_trainValidation)

k = [1,3,5,7] #vector of K 

#Standardization of data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

scaler.fit(X_train) #Compute the mean and std to be used for later scaling.
X_train = scaler.transform(X_train) #Perform standardization by centering and scaling

X_Validation = scaler.transform(X_Validation)

scaler.fit(X_trainValidation)
X_test = scaler.transform(X_test)
X_trainValidation=scaler.transform(X_trainValidation)

from sklearn.neighbors import KNeighborsClassifier
from mlxtend.plotting import plot_decision_regions
import matplotlib.gridspec as gridspec
import itertools

accuracyModel = {} #dict of accuracy values linked to the clfs key:index value:accuracy

model=[] # vectors of the clfs (classifiers)

for i in k:
    model.append((KNeighborsClassifier(n_neighbors=i))) #definition of the clfs

i=0
gs = gridspec.GridSpec(2, 2) #grid 2x2
fig = plt.figure(figsize=(10,8)) # window to visualize the content of the plots

for grd, clf, labels in zip(itertools.product([0, 1], repeat=2),
                        model,
                        ["3-Class classification with KNN(k = 1)","3-Class classification with KNN(k = 3)"
                        ,"3-Class classification with KNN(k = 5)","3-Class classification with KNN(k = 7)"]):
    clf.fit(X_train,y_train)
    ax = plt.subplot(gs[grd[0], grd[1]])

    # Plot the decision boundary. For that, we will assign a color to each
    fig=plot_decision_regions(X_Validation, y_Validation, clf=clf, legend=0)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles,['Wine 1','Wine 2','Wine 3'],framealpha=0.3, scatterpoints=1)
    plt.xlabel("Alcohol")
    plt.ylabel("Malic acid")
    
    accuracyModel[i] = round(model[i].score(X_Validation,y_Validation),3)
    print( f"The accuracy of the model with k equal to {k[i]} is: {accuracyModel[i]}")
    i+=1

plt.show()

#Plot a graph showing how the accuracy on the validation set varies when changing K
accuracylist = sorted(accuracyModel.items()) # sorted by key, return a list of tuples
valueK, accuracy = zip(*accuracylist) # unpack a list of pairs into two tuples objects (key and value)
plt.plot(k, accuracy,'bo--') # plot the accuracy values obtained with respect to the value of K
plt.ylabel('Accuracy level')
plt.xlabel('k')
plt.show()

#pick the best K and evaluate the model performance on the test
bestK = 0
bestV = 0

#search of optimum value for K
for key,value in accuracylist:
    if(value>bestV):
        bestV = value
        bestK = key

model = KNeighborsClassifier(n_neighbors=k[bestK])
model.fit(X_trainValidation, y_trainValidation) 

print(f'The KNN model with k = {k[bestK]} has an accuracy on the test set up to: {round(model.score(X_test,y_test),3)}')
