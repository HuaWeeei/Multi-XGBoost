How to use XGBoost combined with Grid Search for a multi-output regression task?

The following necessary packages will be used:
1、MultiOutputRegressor
2、GridSearchCV
3、Pipeline
4、XGBoost

The main point to note is that within the search range of GridSearchCV, 
additional names need to be filled in so that GridSearch can function within MultiOutputRegressor and Pipeline. 
Finally, XGBoost will be used to predict multiple target values at once.
