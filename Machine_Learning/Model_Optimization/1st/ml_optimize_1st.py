### Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import joblib
import os
##############################           Model Optimization
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_validate,StratifiedKFold,KFold
from sklearn.model_selection import ShuffleSplit ,RepeatedKFold
from sklearn.model_selection import cross_val_score
##############################################################################

### Define current folder path
current_dir = os.path.dirname(__file__)  # __file__ represents the path of the current script
### Define path of training data set
data_path = os.path.join(current_dir, 'smd_training_data_1st.csv')

### Data handing with feature scaling
def data_handling(df_trainning_set):
    X_Train= df_trainning_set.iloc[:,2:7].values
    y_train= df_trainning_set.iloc[:, 7].values
    # Feature Scaling
    sc_X = StandardScaler()
    X_Train = sc_X.fit_transform(X_Train)
    return X_Train,y_train

def feature_importance():
    # Load the saved model
    rf_best = joblib.load('best_rf_model_1st.pkl')
    importance = rf_best.feature_importances_
    # summarize feature importance
    for i,v in enumerate(importance):
        print('Feature: %0d, Score: %.5f' % (i+1,v))
    # plot feature importance
    sns.set(style="whitegrid")
    sns.set_context('paper')
    fig1, ax1 = plt.subplots(figsize=(12,9))
    ax1=plt.bar([x for x in range(len(importance))], importance)
    plt.ylabel('Importance',fontdict={'fontsize':26})
    plt.xlabel('Features',fontdict={'fontsize':26})
    ind=np.arange(5)
    plt.xticks(ind, ('Dissociation Time', 'Displacement', 'Ligand-RMSD', 'Interface-RMSD', '$\Delta$ BSA'))
    plt.xticks(rotation=0, fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    plt.show()
    #############################

def model_optimization(X_Train,y_train):
    model_rf = RandomForestClassifier(random_state=42)
    cv1 = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    #### Parameters
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=1500, stop=2400, num=10)]
    # Criterion
    criterion = ['entropy', 'gini']
    # Number of features to consider at every split
    max_features = [0.5, 1, 2, 3]
    # Maximum number of levels in tree
    max_depth = [20, 25, 30, 35, 40]
    # Minimum number of samples required to split a node
    min_samples_split = [10, 12, 15, 18, 20]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'criterion': criterion,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}
    # Random search of parameters, using 10 fold cross validation,
    rf_random_search = RandomizedSearchCV(estimator=model_rf, param_distributions=random_grid,
                                   n_iter=600, cv=cv1, verbose=2, random_state=42,
                                   n_jobs=-1, return_train_score=True)
    # Fit the random search model
    rf_random_search.fit(X_Train, y_train)
    print("### Turning parameters ###")
    print("Best parameters found: ", rf_random_search.best_params_)
    print("Best cross-validation score: {:.3f}".format(rf_random_search.best_score_))
    best_model = rf_random_search.best_estimator_
    ### Save the best model obtained from random search
    joblib.dump(best_model, 'best_rf_model_1st.pkl')


def main():
    df_trainning_set= pd.read_csv(data_path)
    X_Train, y_train = data_handling(df_trainning_set)
    # model_optimization(X_Train, y_train)
    feature_importance()


if __name__ == '__main__':
    main()





