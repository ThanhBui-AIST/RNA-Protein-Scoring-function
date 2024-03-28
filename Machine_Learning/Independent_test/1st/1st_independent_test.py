# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import joblib
import os
##############################################################################
### Define current folder path
current_dir = os.path.dirname(__file__)  # __file__ represents the path of the current script
### Define the relative path to the file in current folder
### Define path of training data set
data_path = os.path.join(current_dir, '../../Model_Optimization/1st/smd_training_data_1st.csv')
### Define path of Testing data set
data_test_path=os.path.join(current_dir, 'testing_data_1st.csv')
### Define path of model checkpointing
model_path= os.path.join(current_dir, '../../Model_Optimization/1st/best_rf_model_1st.pkl')

### Data handing with feature scaling
def data_handling(df_trainning_set,df_testing_set):
    X_Train= df_trainning_set.iloc[:,2:7].values
    Y_Train= df_trainning_set.iloc[:, 7].values
    X_Test = df_testing_set.iloc[:,2:7].values
    Y_Test = df_testing_set.iloc[:, 7].values
    # Feature Scaling
    sc_X = StandardScaler()
    X_Train = sc_X.fit_transform(X_Train)
    X_Test = sc_X.transform(X_Test)
    return X_Test, Y_Test

### Show prediction of each pose in SMD simulation
def result_each_pose(y_test,y_pred):
    #number of replicacs
    num=3
    col=round(len(y_pred)/num)
    A1=np.reshape(y_pred,(col,num))
    A2=np.reshape(y_test,(col,num))
    Predict, Correct =[],[]
    for i in range (col):
        Predict.append(float(np.bincount(A1[i,:]).argmax()))
        Correct.append(float(np.bincount(A2[i,:]).argmax()))
    bool_list = list(map(lambda x, y: x == y, Predict, Correct))
    print('Comparing correct answers for each structures :',bool_list)
    count=np.bincount(bool_list)
    print('Number of poses : ', col)
    print('[Fail  Correct] :',count)


### Making the Confusion Matrix ##
def plot_data(y_test,y_pred):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax1 = plt.subplots(figsize=(8.5,8))
    ax1=sns.heatmap(cm, annot=True,cbar=False,
                cmap='Set3',annot_kws={'size':50},fmt=".0f")
    ax1.set_ylabel('True class',fontdict={'weight': 'bold','fontsize':28})
    ax1.set_xlabel('Predicted class',fontdict={'weight': 'bold','fontsize':28})
    ax1.xaxis.set_tick_params(labelsize=28)
    ax1.yaxis.set_tick_params(labelsize=28)
    plt.tight_layout()
    # plt.savefig('1st_round_it.tiff',dpi=300)
    plt.show()


def main():
    df_trainning_set= pd.read_csv(data_path)
    df_testing_set= pd.read_csv(data_test_path)
    X_Test, y_test = data_handling(df_trainning_set,df_testing_set)
    # Loading Model Checkpointing
    model_rf = joblib.load(model_path)
    # Predicting the test set results
    y_pred = model_rf.predict(X_Test)
    print("Random Forest Score :",model_rf.score(X_Test,y_test))
    result_each_pose(y_test, y_pred)
    plot_data(y_test, y_pred)


if __name__ == '__main__':
    main()