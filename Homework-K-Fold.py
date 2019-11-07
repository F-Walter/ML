from sklearn.datasets import load_wine
X, y = load_wine(return_X_y=True)


import matplotlib.pyplot as plt
import numpy as np
X_slice =  X[:,:2]
X_plot = np.transpose(X_slice)


plt.scatter(X_plot[0],X_plot[1],c=y)
plt.title("Plot of the data according to the class:")
plt.xlabel("Alcohol")
plt.ylabel("Malic acid")
plt.show()


from sklearn.model_selection import train_test_split
X_trainValidation,X_test, y_trainValidation, y_test = train_test_split(X_slice, y,test_size=0.30,random_state=1,stratify=y)

# Standardization of data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_trainValidation)
X_trainValidation = scaler.transform(X_trainValidation)
X_test = scaler.transform(X_test)

from sklearn.svm import SVC
accuracyModel = {} #dict key=C value=accuracyModel

C = [0.001, 0.01, 0.1, 1, 10, 100,1000]
gamma = [10**(-11),10**(-9),10**(-7),10**(-5),10**(-3),10**(-1),10]

'''
15. Perform a grid search of the best parameters for an RBF kernel: we will
now tune both gamma and C at the same time. Select an appropriate
range for both parameters. Train the model and score it on the validation
set. Evaluate the best parameters on the test set. Plot the decision
boundaries.
'''
from sklearn.model_selection import GridSearchCV
from mlxtend.plotting import plot_decision_regions
param_grid={
'C': C,
'kernel': ['rbf'],
'gamma': gamma
}

model = GridSearchCV(SVC(),param_grid,n_jobs=-1,iid=True,cv=5)
model.fit(X_trainValidation,y_trainValidation)

print(f'The model was created with C = {model.best_params_["C"]} and gamma = {model.best_params_["gamma"]}')

print(f'The model scored on the test an accuracy equal to: {model.score(X_test,y_test)}')

fig = plt.figure(figsize=(25,25))
fig = plot_decision_regions(X=X_test,y=y_test,clf=model,legend=2)

plt.title(f"3-Class classification with SVM(C= {model.best_params_['C']} and gamma = {model.best_params_['gamma']})")
plt.xlabel("Alcohol")
plt.ylabel("Malic acid")
plt.show()   