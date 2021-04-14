
#--------------------------- MODEL TRAINING ----------------------------------

from sklearn.metrics import r2_score,mean_squared_error

accuracyScores=[]
accuracyScores2=[]
print('\n----------- SVM ------------')
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train,y_train)
y_pred8 = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred8)
#print(cm)
print('Score on train data: {}'.format(classifier.score(X_train,y_train)))
print('Score on test data: {}\n'.format(classifier.score(X_test,y_test)))
accuracy =accuracy_score(y_test,y_pred8)
print('Overall model accuracy: {}'.format(accuracy))
accuracyScores.append(accuracy)

print('\n----------- SVM Kernel ------------')
classifierKernel = SVC(kernel = 'rbf', random_state = 0)
classifierKernel.fit(X_train, y_train)
print('Score on train data: {}'.format(classifierKernel.score(X_train,y_train)))
print('Score on test data: {}\n'.format(classifierKernel.score(X_test,y_test)))
y_pred7 = classifierKernel.predict(X_test)
accuracy = accuracy_score(y_test,y_pred7)
print('Overall model accuracy: {}'.format(accuracy))
accuracyScores.append(accuracy)

print('\n----------- GaussianNB ------------')
from sklearn.naive_bayes import GaussianNB
classifierNB = GaussianNB()
classifierNB.fit(X_train, y_train)
print('Score on train data: {}'.format(classifierNB.score(X_train,y_train)))
print('Score on test data: {}\n'.format(classifierNB.score(X_test,y_test)))
y_pred6 = classifierNB.predict(X_test)
accuracy = accuracy_score(y_test,y_pred6)
print('Overall model accuracy: {}'.format(accuracy))
accuracyScores.append(accuracy)

print('\n----------- K-NN ------------')
from sklearn.neighbors import KNeighborsClassifier
classifierKNN = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifierKNN.fit(X_train, y_train)
print('Score on train data: {}'.format(classifierKNN.score(X_train,y_train)))
print('Score on test data: {}\n'.format(classifierKNN.score(X_test,y_test)))
y_pred5 = classifierKNN.predict(X_test)
accuracy = accuracy_score(y_test,y_pred5)
print('Overall model accuracy: {}'.format(accuracy))
accuracyScores.append(accuracy)

print('\n----------- Logistic Regression ------------')
from sklearn.linear_model import LogisticRegression
classifierLR = LogisticRegression(random_state = 0)
classifierLR.fit(X_train, y_train)
print('Score on train data: {}'.format(classifierLR.score(X_train,y_train)))
print('Score on test data: {}\n'.format(classifierLR.score(X_test,y_test)))
y_pred4 = classifierLR.predict(X_test)
accuracy = accuracy_score(y_test,y_pred4)
print('Overall model accuracy: {}'.format(accuracy))
accuracyScores.append(accuracy)

print('\n----------- Decision Tree  ------------')
from sklearn.tree import DecisionTreeClassifier
classifierDT = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifierDT.fit(X_train, y_train)
print('Score on train data: {}'.format(classifierDT.score(X_train,y_train)))
print('Score on test data: {}\n'.format(classifierDT.score(X_test,y_test)))
y_pred3 = classifierDT.predict(X_test)
accuracy =accuracy_score(y_test,y_pred3)
print('Overall model accuracy: {}'.format(accuracy))
accuracyScores.append(accuracy)

print('\n----------- Decision Tree Grid Search ------------')
from sklearn.model_selection import GridSearchCV,train_test_split
params = {'max_depth':[3,4,5,6,7],
         'max_features':['auto','sqrt','log2'],
         'min_samples_split':[2,3,4,5,6,7,8,9,10],
         'min_samples_leaf':[2,3,4,5,6,7,8,9,10]}
'''
tree = DecisionTreeClassifier()
tree_search = GridSearchCV(tree,param_grid=params,n_jobs=-1,cv=5)
tree_search.fit(X_train,y_train)
print('Score on train data: {}'.format(tree_search.score(X_train,y_train)))
print('Score on test data: {}'.format(tree_search.score(X_test,y_test)))
print('Best parameters found:')
display(tree_search.best_params_)
tree_search_pred = tree_search.predict(X_test)
tree_search_accuracy = accuracy_score(y_test,tree_search_pred)
print('Overall model accuracy: {}'.format(tree_search_accuracy))
'''

print('\n----------- Random Forest  ------------')
from sklearn.ensemble import RandomForestClassifier
classifierRF = RandomForestClassifier(n_estimators = 1000, criterion = 'entropy', random_state = 0)
classifierRF.fit(X_train, y_train)
print('Score on train data: {}'.format(classifierRF.score(X_train,y_train)))
print('Score on test data: {}\n'.format(classifierRF.score(X_test,y_test)))
y_pred2 = classifierRF.predict(X_test)
accuracy = accuracy_score(y_test,y_pred2)
print('Overall model accuracy: {}'.format(accuracy))
accuracyScores.append(accuracy)

print('\n------------------ MLP  ----------------')
from sklearn.neural_network import MLPClassifier
clfMLP = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
clfMLP.fit(X_train, y_train)
print('Score on train data: {}'.format(clfMLP.score(X_train,y_train)))
print('Score on test data: {}\n'.format(clfMLP.score(X_test,y_test)))
y_pred2 = clfMLP.predict(X_test)
accuracy = accuracy_score(y_test,y_pred2)
print('Overall model accuracy: {}'.format(accuracy))
accuracyScores.append(accuracy)
print('Parameters currently in use:\n')
print(clfMLP.get_params())

print('\n----------- GradientBoostingClassifier  ------------')
from sklearn.ensemble import GradientBoostingClassifier
clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1,
                                 max_depth=1, random_state=0).fit(X_train, y_train)
clf.fit(X_train, y_train)
print('Score on train data: {}'.format(clf.score(X_train,y_train)))
print('Score on test data: {}\n'.format(clf.score(X_test,y_test)))
y_pred1 = clf.predict(X_test)
accuracy = accuracy_score(y_test,y_pred1)
print('Overall model accuracy: {}'.format(accuracy))
accuracyScores.append(accuracy)


print('\n----------- RandomizedSearchCV  ------------')
from sklearn.model_selection import RandomizedSearchCV
params['learning_rate'] = np.linspace(0.1,0.2,10)#start,stop, num
clf1 = GradientBoostingClassifier()
clf = RandomizedSearchCV(clf1,params,n_jobs=-1,cv=5,verbose=2)
clf.fit(X_train, y_train)
print('Score on train data: {}'.format(clf.score(X_train,y_train)))
print('Score on test data: {}\n'.format(clf.score(X_test,y_test)))
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test,y_pred)
print('Overall model accuracy: {}'.format(accuracy))
accuracyScores.append(accuracy)
accuracyScores2.append(accuracy)
print(clf.get_params())
print('Best parameters found:')
display(clf.best_params_)


print('\n-------------------------- K-FOLD CROSS VALIDATION  ------------------------------------')
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler

print('\n----------- SVM ------------')
scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(X)

scores = []
best_model= SVC(kernel = 'linear', random_state = 0)
cv = KFold(10, True,1)
for train_index, test_index in cv.split(X):
    #print("Train Index: ", train_index, "\n")
    #print("Test Index: ", test_index)
    X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
    best_model.fit(X_train, y_train)
    print('Score on test data: {}\n'.format(best_model.score(X_test,y_test)))
    scores.append(best_model.score(X_test, y_test))
print("MESOS OROS ACCURACY: ", np.mean(scores)) 
accuracyScores2.append(np.mean(scores))


print('\n----------- SVM rbf------------')
scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(X)

scores = []
best_model= SVC(kernel = 'rbf', random_state = 0)
cv = KFold(10, True,1)
for train_index, test_index in cv.split(X):
    #print("Train Index: ", train_index, "\n")
    #print("Test Index: ", test_index)
    X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
    best_model.fit(X_train, y_train)
    y_pred77 = best_model.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred77)
    print('Overall model accuracy: {}'.format(accuracy))
    print('Score on test data: {}\n'.format(best_model.score(X_test,y_test)))
    scores.append(best_model.score(X_test, y_test))
print("MESOS OROS ACCURACY: ", np.mean(scores)) 
accuracyScores2.append(np.mean(scores))

print('\n----------- Gaussian naive bayes ------------')
scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(X)
scores = []
best_model= GaussianNB()
cv = KFold(10, True,1)
for train_index, test_index in cv.split(X):
    #print("Train Index: ", train_index, "\n")
    #print("Test Index: ", test_index)
    X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
    best_model.fit(X_train, y_train)
    print('Score on test data: {}\n'.format(best_model.score(X_test,y_test)))
    scores.append(best_model.score(X_test, y_test))
print("MESOS OROS ACCURACY: ", np.mean(scores))
accuracyScores2.append(np.mean(scores))

print('\n----------- Logistic Regression ------------')
scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(X)
scores = []
best_model= LogisticRegression(random_state = 0)
cv = KFold(10, True,1)
for train_index, test_index in cv.split(X):
    #print("Train Index: ", train_index, "\n")
    #print("Test Index: ", test_index)
    X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
    best_model.fit(X_train, y_train)
    print('Score on test data: {}\n'.format(best_model.score(X_test,y_test)))
    scores.append(best_model.score(X_test, y_test))
print("MESOS OROS ACCURACY: ", np.mean(scores))
accuracyScores2.append(np.mean(scores))

print('\n----------- MLP ------------')
scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(X)
scores = []
best_model2= MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
cv = KFold(10, True,1)
for train_index, test_index in cv.split(X):
    #print("Train Index: ", train_index, "\n")
    #print("Test Index: ", test_index)
    X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
    best_model2.fit(X_train, y_train)
    print('Score on test data: {}\n'.format(best_model2.score(X_test,y_test)))
    scores.append(best_model2.score(X_test, y_test))
print("MESOS OROS ACCURACY: ", np.mean(scores))
accuracyScores2.append(np.mean(scores))

print('\n----------- KNN ------------')
scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(X)

scores = []
best_model= KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
cv = KFold(10, True,1)
for train_index, test_index in cv.split(X):
    #print("Train Index: ", train_index, "\n")
    #print("Test Index: ", test_index)
    X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
    best_model.fit(X_train, y_train)
    print('Score on test data: {}\n'.format(best_model.score(X_test,y_test)))
    scores.append(best_model.score(X_test, y_test))
print("MESOS OROS ACCURACY: ", np.mean(scores)) 
accuracyScores2.append(np.mean(scores))

print('\n----------- Decision trees ------------')
scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(X)

scores = []
#best_model= RandomForestClassifier(n_estimators = 15, criterion = 'entropy', random_state = 0)
best_model = DecisionTreeClassifier( criterion = 'entropy', random_state = 0)
cv = KFold(10, True,1)
for train_index, test_index in cv.split(X):
    #print("Train Index: ", train_index, "\n")
    #print("Test Index: ", test_index)
    X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
    best_model.fit(X_train, y_train)
    print('Score on test data: {}\n'.format(best_model.score(X_test,y_test)))
    scores.append(best_model.score(X_test, y_test))
print("MESOS OROS ACCURACY: ", np.mean(scores))  
accuracyScores2.append(np.mean(scores))

print('\n----------- Random forests ------------')
scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(X)

scores = []
best_model1= RandomForestClassifier(n_estimators = 15, criterion = 'entropy', random_state = 0)

cv = KFold(10, True,1)
for train_index, test_index in cv.split(X):
    #print("Train Index: ", train_index, "\n")
    #print("Test Index: ", test_index)
    X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
    best_model1.fit(X_train, y_train)
    print('Score on test data: {}\n'.format(best_model1.score(X_test,y_test)))
    scores.append(best_model1.score(X_test, y_test))
print("MESOS OROS ACCURACY: ", np.mean(scores)) 
accuracyScores2.append(np.mean(scores))


print('\n----------- Gradient boosting classifier ------------')
scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(X)

scores = []
best_model= GradientBoostingClassifier(n_estimators=100, learning_rate=1,
                                 max_depth=1, random_state=0).fit(X_train, y_train)

cv = KFold(10, True,1)
for train_index, test_index in cv.split(X):
    #print("Train Index: ", train_index, "\n")
    #print("Test Index: ", test_index)
    X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
    best_model.fit(X_train, y_train)
    print('Score on test data: {}\n'.format(best_model.score(X_test,y_test)))
    scores.append(best_model.score(X_test, y_test))
print("MESOS OROS ACCURACY: ", np.mean(scores)) 
accuracyScores2.append(np.mean(scores)) 
 
from sklearn.model_selection import RepeatedStratifiedKFold  

print('\n----------- BaggingClassifier GridSearch  ------------')
from sklearn.ensemble import BaggingClassifier
n_estimators = [10, 100, 1000]
model = BaggingClassifier()     #by deafult estimator einai to decisiontreeclassifier
# define grid search
grid = dict(n_estimators=n_estimators)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
grid_result = grid_search.fit(X_train, y_train)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param)) 
accuracyScores.append(grid_result.best_score_)    
accuracyScores2.append(grid_result.best_score_) 

#kanw tis times twn score se pososto 
new_accuracyScores = [round(i*100) for i in accuracyScores]
new_accuracyScores2 = [round(i*100) for i in accuracyScores2]

################################ SCORES ######################################
print("\n ------------- Score of Models -----------------")

modelList=['SVM','SVM Kernel','Naive Bayes','K-NN','Logistic Regression','Decision Tree',
           'Random Forest','MLP','Gradient Boosting Classifier','RandomizedSearchCV',
           'GridSearch Bagging Classifier']

for i,j in zip(modelList, new_accuracyScores):
    print(i,'Score:',j,"%")
#plot the score of models        
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.barh(modelList,new_accuracyScores,color=['mediumseagreen'])
plt.title('Score of models')
plt.xlabel('Score')
for modelList, value in enumerate(new_accuracyScores):
    plt.text(value, modelList, str(value))
plt.show()



print("\n ------------- Score of Models with K -----------------")
modelList2=['Randomized search','SVM ','SVM rbf','Naive Bayes','Logistic Regression','MLP','K-NN','Decision Tree',
           'Random Forest','Gradient Boosting Classifier','Bagging Classifier']

for i,j in zip(modelList2, new_accuracyScores2):
    print(i,'Score:',j,"%")
#plot the score of models        
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.barh(modelList2,new_accuracyScores2,color=['mediumseagreen'])
plt.title('Score of models')
plt.xlabel('Score')
for modelList2, value in enumerate(new_accuracyScores2):
    plt.text(value, modelList2, str(value))
plt.show()
