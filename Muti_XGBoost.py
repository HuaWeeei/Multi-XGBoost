# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 15:34:28 2022

@author: bbill
"""
from tensorflow.keras.models import Model,load_model
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense,LeakyReLU,BatchNormalization
from tensorflow.keras.utils import plot_model
import xgboost as xgb
import time
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
df = pd.read_csv("2022-train-v2.csv")
df = df.fillna(0)
X=df.iloc[:,6:131]
yy = df.iloc[:,0:6] 
coX= X
train_X_s = StandardScaler()     
X = train_X_s.fit_transform(X)
def ss(a):
    X = train_X_s.fit_transform(a)
    return X
train_y_s = StandardScaler()
yy = pd.DataFrame(yy)
yy = train_y_s.fit_transform(yy)
train_X, test_X, train_y, test_y = train_test_split( X, yy,test_size=0.3)

n_inputs = train_X.shape[1]
n_bottleneck=(round(float(n_inputs)/5.0))
visible=Input(shape=(n_inputs,),name='Input-Layer')
e = Dense(units=n_inputs,name='Encoder-Layer')(visible)
e = BatchNormalization(name='Encoder-Layer-Normalization')(e)
e = LeakyReLU(name='Encoder-Layer-Activation')(e)

bottleneck = Dense(units=n_bottleneck, name='Bottleneck-Layer')(e)
d = Dense(units=n_inputs, name='Decoder-Layer')(bottleneck)
d = BatchNormalization(name='Decoder-Layer-Normalization')(d)
d = LeakyReLU(name='Decoder-Layer-Activation')(d)

output = Dense(units=n_inputs, activation='linear', name='Output-Layer')(d)
model = Model(inputs=visible, outputs=output, name='Autoencoder-Model')
model.compile(optimizer='adam', loss='mse')
history = model.fit(train_X, train_X, epochs=10, batch_size=16, verbose=1, validation_data=(test_X, test_X))
encoder = Model(inputs=visible,outputs=bottleneck)
encoder.compile(optimizer='adam',loss='mse')
encoder.save('DeleteOrA_1')  
encoder = load_model('AE_A3_Y') 
AEtr_p = encoder.predict(X)
tStart_m = time.time()
xgb_grid = xgb.XGBRegressor(objective = 'reg:squarederror',colsample_bytree=1, eta=0.3, gamma= 0.1, 
                            max_depth= 6,n_estimators= 1000,subsample= 0.8)
'''XGBoost with GridSearch in Muti-Tasks'''
regress_model = xgb_grid
pipeline = Pipeline([('reg', MultiOutputRegressor(regress_model))])
cv_params = {'reg__estimator__n_estimators':[500]
                 ,'reg__estimator__eta':[0.2,0.3]
                  ,'reg__estimator__gamma':[0.1,0.2]
                 ,'reg__estimator__max_depth':[3,4,5],'reg__estimator__subsample':[0.7,0.9]
                , 'reg__estimator__colsample_bytree':[0.6,0.8],'reg__estimator__min_child_weight':[0.4,0.5,0.6]
                            }
from sklearn.metrics import make_scorer
def rmse_sco(predict, actual):   #評估gridsearchcv的分數
    predict = np.array(predict)
    actual = np.array(actual)
    distance = predict - actual
    square_distance = distance ** 2
    mean_square_distance = square_distance.mean()
    score = np.sqrt(mean_square_distance)
    return score
r_rmse = make_scorer(rmse_sco,greater_is_better=(False)) #a smaller RMSE is better, fill in "false
gs_m = GridSearchCV(pipeline , cv_params, verbose=2
                   ,refit=True, n_jobs=1,scoring=r_rmse 
                  ,cv=5)
gs_m.fit(AEtr_p,yy)
gs_p = gs_m.best_estimator_
a =gs_m.best_params_
Params=[]
Params.append(a)
# joblib.dump(gs_p, 'DeleteXGBA.model') 
tEnd_m = time.time()
run = 1
turn = 2
AverageRmse=[]
AverageMape=[]
AverageTime=[]
while run <= turn :
    train_X, test_X, train_y, test_y = train_test_split( X, yy,test_size=0.3)
    AEte_p = encoder.predict(test_X)
    p = gs_p.predict(AEte_p)
    test_y = train_y_s.inverse_transform(test_y)  
    # p = np.array(p).reshape(-1,1) #maybe need reshape
    p = train_y_s.inverse_transform(p)
    def inverse(a):
        pre = train_y_s.inverse_transform(a)
        return pre
    test_y = pd.DataFrame(test_y)
    p = pd.DataFrame(p)
    tStart_v = time.time()
    def rmse(o_r,p_r):
        n = len(o_r)
        mse = np.sum((o_r-p_r)**2)/(n)
        armse = np.sqrt(mse)
        return armse
    def mape(o_r,p_r):
        n = len(o_r)
        ma = (sum(np.abs((o_r-p_r)/o_r)))/n*100
        return ma
    dy_1 = test_y.iloc[:,0:1]
    dp_1 = p.iloc[:,0:1]
    dy_2= test_y.iloc[:,1:2]
    dp_2 = p.iloc[:,1:2]
    dy_3 = test_y.iloc[:,2:3]
    dp_3 = p.iloc[:,2:3]
    dy_4 = test_y.iloc[:,3:4]
    dp_4 = p.iloc[:,3:4]
    dy_5 = test_y.iloc[:,4:5]
    dp_5 = p.iloc[:,4:5]
    dy_6 = test_y.iloc[:,5:6]
    dp_6 = p.iloc[:,5:6]
    coX_c = coX.columns
    coX_c = coX_c[0:n_bottleneck]
    test_X = pd.DataFrame(AEte_p,columns = coX_c)
    test_X['o_1']= dy_1
    test_X['p_1']= dp_1
    test_X['o_2']= dy_2
    test_X['p_2']= dp_2
    test_X['o_3']= dy_3
    test_X['p_3']= dp_3
    test_X['o_4']= dy_4
    test_X['p_4']= dp_4
    test_X['o_5']= dy_5
    test_X['p_5']= dp_5
    test_X['o_6']= dy_6
    test_X['p_6']= dp_6
    rmse1 =rmse(test_X['o_1'],test_X['p_1'])
    rmse2 =rmse(test_X['o_2'],test_X['p_2'])
    rmse3 =rmse(test_X['o_3'],test_X['p_3'])
    rmse4 =rmse(test_X['o_4'],test_X['p_4'])
    rmse5 =rmse(test_X['o_5'],test_X['p_5'])
    rmse6 =rmse(test_X['o_6'],test_X['p_6'])
    Av = (rmse1+rmse2+rmse3+rmse4+rmse5+rmse6)/6
    
    mape1 =mape(test_X['o_1'],test_X['p_1'])
    mape2 =mape(test_X['o_2'],test_X['p_2'])
    mape3 =mape(test_X['o_3'],test_X['p_3'])
    mape4 =mape(test_X['o_4'],test_X['p_4'])
    mape5 =mape(test_X['o_5'],test_X['p_5'])
    mape6 =mape(test_X['o_6'],test_X['p_6'])
    Am = (mape1+mape2+mape3+mape4+mape5+mape6)/6
    df_r = pd.DataFrame({'rmse1':[rmse1],'rmse2':[rmse2]
                         ,'rmse3':[rmse3],'rmse4':[rmse4],'rmse5':[rmse5],'rmse6':[rmse6]})
    df_m = pd.DataFrame({'mape1':[mape1],'mape2':[mape2]
                         ,'mape3':[mape3],'mape4':[mape4],'mape5':[mape5],'mape6':[mape6]})
    AverageRmse.append(Av)
    AverageMape.append(Am)
    run += 1
averrmse = np.mean(AverageRmse)
avermape = np.mean(AverageMape)
avertime = np.mean(AverageTime)
stdrmse = np.std(AverageRmse)
stdrmape = np.std(AverageMape)
print('Params：',Params)
print('AverageRmse：',averrmse)
print('AverageMape：',avermape)
print('StdMape：',stdrmape)
print('StdRmse：',stdrmse)
b = pd.DataFrame(AverageRmse)
c = pd.DataFrame(AverageMape)
o = pd.concat([b,c],axis = 1)
col =['Rmse','Mape']
o.columns = col
o.to_csv('Predict_Result.csv',index = False)
