#--------------------------- MODEL TRAINING ----------------------------------

from sklearn.linear_model import LinearRegression
lin_model = LinearRegression()
lin_model.fit(X_train,y_train) #fit the model

print('------L Regression -------')
print('Score on train data: {}'.format(lin_model.score(X_train,y_train)))
print('Score on test data: {}\n'.format(lin_model.score(X_test,y_test)))
from sklearn.metrics import r2_score,mean_squared_error
prediction = lin_model.predict(X_test)
mse = mean_squared_error(y_test,prediction)
accuracy = r2_score(y_test,prediction)

print('Mean Squared Error: {}'.format(mse))
print('Overall model accuracy: {}'.format(accuracy))

#2 Ensemble methods
from sklearn.tree import DecisionTreeRegressor

decision_tree = DecisionTreeRegressor(max_depth=5,max_features='auto',min_samples_split=3, min_samples_leaf=2)
decision_tree.fit(X_train,y_train)

print('\n----------- Decision Tree Regressor ------------')
print('Score on train data: {}'.format(decision_tree.score(X_train,y_train)))
print('Score on test data: {}\n'.format(decision_tree.score(X_test,y_test)))

tree_pred = decision_tree.predict(X_test)
tree_mse = mean_squared_error(y_test,tree_pred)
tree_accuracy = r2_score(y_test,tree_pred)

print('Root Mean Squared Error: {}'.format(np.sqrt(tree_mse)))
print('Overall model accuracy: {}'.format(tree_accuracy))

from sklearn.model_selection import GridSearchCV,train_test_split
print('\n----------- GridSearch ------------')

params = {'max_depth':[3,4,5,6,7],
         'max_features':['auto','sqrt','log2'],
         'min_samples_split':[2,3,4,5,6,7,8,9,10],
         'min_samples_leaf':[2,3,4,5,6,7,8,9,10]}

tree = DecisionTreeRegressor()

tree_search = GridSearchCV(tree,param_grid=params,n_jobs=-1,cv=5)

tree_search.fit(X_train,y_train)   # fit the model


print('Score on train data: {}'.format(tree_search.score(X_train,y_train)))
print('Score on test data: {}'.format(tree_search.score(X_test,y_test)))
print('Best parameters found:')
display(tree_search.best_params_)

tree_search_pred = tree_search.predict(X_test)
tree_search_mse = mean_squared_error(y_test,tree_search_pred)
tree_search_accuracy = r2_score(y_test,tree_search_pred)

print('Root Mean Squared Error: {}'.format(np.sqrt(tree_search_mse)))
print('Overall model accuracy: {}'.format(tree_search_accuracy))


# RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor
forest_search = RandomForestRegressor(n_estimators=100,max_depth=7, max_features='auto',
                              min_samples_split=7,min_samples_leaf=3)
forest_search.fit(X_train,y_train)
print('\n----------- RandomForest ------------')
print('\nScore on train data: {}'.format(forest_search.score(X_train,y_train)))
print('Score on test data: {}'.format(forest_search.score(X_test,y_test)))

forest_search_pred = forest_search.predict(X_test)
forest_search_mse = mean_squared_error(y_test,forest_search_pred)
forest_search_accuracy = r2_score(y_test,forest_search_pred)

print('Root Mean Squared Error: {}'.format(np.sqrt(forest_search_mse)))
print('Overall model accuracy: {}'.format(forest_search_accuracy))


#MLP
from sklearn.neural_network import MLPRegressor
clfMLP = MLPRegressor(hidden_layer_sizes=(64,64,64),activation="relu" ,random_state=1, max_iter=2000)
clfMLP.fit(X_train, y_train)
print('\n----------- MLP ------------')
print('\nScore on train data: {}'.format(clfMLP.score(X_train,y_train)))
print('Score on test data: {}'.format(clfMLP.score(X_test,y_test)))
MLP_search_pred = forest_search.predict(X_test)
MLP_search_mse = mean_squared_error(y_test,forest_search_pred)
MLP_search_accuracy = r2_score(y_test,forest_search_pred)

print('Root Mean Squared Error: {}'.format(np.sqrt(MLP_search_mse)))
print('Overall model accuracy: {}'.format(MLP_search_accuracy))


#GradientBoostingRegressor
from sklearn.ensemble import  GradientBoostingRegressor
grad_boost = GradientBoostingRegressor(n_estimators=100,max_depth=7,max_features='auto',
                                      min_samples_split=7,min_samples_leaf=3,learning_rate=0.1)
grad_boost.fit(X_train,y_train)
print('\n----------- GradientBoostingRegressor ------------')
print('Score on train data: {}'.format(grad_boost.score(X_train,y_train)))
print('Score on test data: {}'.format(grad_boost.score(X_test,y_test)))

gboost_pred = grad_boost.predict(X_test)
gboost_mse = mean_squared_error(y_test,gboost_pred)
gboost_accuracy = r2_score(y_test,gboost_pred)

print('Root Mean Squared Error: {}'.format(np.sqrt(gboost_mse)))
print('Overall model accuracy: {}'.format(gboost_accuracy))

from sklearn.model_selection import RandomizedSearchCV


params['learning_rate'] = np.linspace(0.1,0.2,10)#start,stop, num

# instantiate the model
gradient_boosting = GradientBoostingRegressor()

# perform the grid search for the best parameters
print('\n-----------G Boost grid search  ------------')
gboost_search = RandomizedSearchCV(gradient_boosting,params,n_jobs=-1,
                                   cv=5,verbose=2)
gboost_search.fit(X_train,y_train)

print('Score on train data: {}'.format(gboost_search.score(X_train,y_train)))
print('Score on test data: {}'.format(gboost_search.score(X_test,y_test)))
print('Best parameters found:')
display(gboost_search.best_params_)

gboost_search_pred = gboost_search.predict(X_test)
gboost_search_mse = mean_squared_error(y_test,gboost_search_pred)
gboost_search_accuracy = r2_score(y_test,gboost_search_pred)

print('Root Mean Squared Error: {}'.format(np.sqrt(gboost_search_mse)))
print('Overall model accuracy: {}'.format(gboost_search_accuracy))

    
predicted=gboost_search.predict(X)

#parity plot
fig, ax = plt.subplots()
ax.scatter(y,predicted)
#ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()


