import abc
import numpy as np
from  scipy import optimize
import pandas as pd
from time import time
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score


from TS_Extrinsic_Regression.models.rocket import generate_kernels,apply_kernels

def get_best_lags_kernels(min_lag,max_lag,kernels_list,y,train_size):
    y_train, y_test=y[:train_size],y[train_size:]
    results_time=[]
    results_r2=[]
    for num_lags in np.arange(min_lag,max_lag):
        for n in kernels_list:
            X_train=pd.concat([pd.DataFrame(y_train).shift(i) for i in range(1,num_lags+1)],axis=1).dropna()
            np.random.seed=13
    
            X_train.columns=[f'train_lag_{i}' for i in range(1,num_lags+1)]

            X_test=pd.concat([pd.DataFrame(y_test).shift(i) for i in range(1,num_lags+1)],axis=1).dropna()
            X_test.columns=[f'train_lag_{i}' for i in range(1,num_lags+1)]
            #X_test=pd.([[pd.Series(X_test.iloc[i,:])] for i in range(X_test.shape[0])])

            tr1=np.expand_dims(X_train.to_numpy(),axis=2)
            test1=np.expand_dims(X_test.to_numpy(),axis=2)

            t0=time()
            kernels = generate_kernels(tr1.shape[1], n, tr1.shape[2])
            x_training_transform = apply_kernels(tr1, kernels)
            x_test_transform = apply_kernels(test1, kernels)
            results_time.append([n,num_lags,time()-t0])

            best_ridge = RidgeCV(alphas=np.logspace(-10, 10, 10), normalize=False,cv=3)

            best_ridge.fit(x_training_transform, y_train[num_lags:])

            predictions=best_ridge.predict(x_test_transform)
            results_r2.append([n,num_lags,r2_score(y_test[num_lags:],predictions)])
    return results_r2,results_time

def h_step_forecast(num_kernels,num_lags,y,train_size,h):
    
    y_train, y_test=y[:train_size],y[train_size:]
    X_train=pd.concat([pd.DataFrame(y_train).shift(i) for i in range(1,num_lags+1)],axis=1).dropna()
    X_train.columns=[f'train_lag_{i}' for i in range(1,num_lags+1)]
    X_test=pd.concat([pd.DataFrame(y_test).shift(i) for i in range(1,num_lags+1)],axis=1).dropna()
    X_test.columns=[f'train_lag_{i}' for i in range(1,num_lags+1)]

    tr1=np.expand_dims(X_train.to_numpy(),axis=2)
    test1=np.expand_dims(X_test.to_numpy(),axis=2)


    kernels = generate_kernels(tr1.shape[1], num_kernels, tr1.shape[2])
    x_training_transform = apply_kernels(tr1, kernels)
    x_test_transform = apply_kernels(test1, kernels)


    best_ridge = RidgeCV(alphas=np.logspace(-10, 10, 10), normalize=False,cv=3)
    best_ridge.fit(x_training_transform, y_train[num_lags:])

    predictions=best_ridge.predict(x_test_transform)
    X_test_h_step_pred=X_test.copy()
    pred_df=pd.DataFrame(predictions)
    pred_df.index=X_test.index
    pred_df.columns=[f'pred t+{1}']

    X_test_h_step_pred=pd.concat([pred_df,X_test_h_step_pred],axis=1)
    X_test_h_step_pred=X_test_h_step_pred.iloc[:,:-1]
    
    for i in range(2,h+1):
        test_h=np.expand_dims(X_test_h_step_pred.to_numpy(),axis=2)

        x_test_transform_h_step = apply_kernels(test_h, kernels)

        predictions=best_ridge.predict(x_test_transform_h_step)

        pred_df=pd.DataFrame(predictions)
        pred_df.index=X_test.index
        pred_df.columns=[f'pred t+{i}']

        X_test_h_step_pred=pd.concat([pred_df,X_test_h_step_pred],axis=1)
        X_test_h_step_pred=X_test_h_step_pred.iloc[:,:-1]
    
    return X_test_h_step_pred


def fit_predict(df_train, df_test, num_lags=2, time=True, lag=True, name='Something'):
    # Extraction of date variables for train set
    X, Y = change_df(y, num_lags, time, lag)
    # Creation of grid to find the best params
    grid = {'n_estimators':[100,500,1000,2000], 'max_depth':[1,2,3,5], 
            'max_leaves':[1,2,5,10], 'n_jobs':[-1]}
    #grid={'max_depth':[1,2,3,5]}
    
    # Creation of GridSearchCV with custom cross validation folders
    # Folders were visualized earlier
    model = GridSearchCV(XGBRegressor(), param_grid=grid, cv=cv, verbose=1)
    
    # Train of the model on train dataset
    model.fit(X, Y)
    
    # Predicting new values step by step:
    x_test, y_test = change_df(y_validate, num_lags, time, lag)
    pred = model.predict(x_test)
    
    fig,ax = plt.subplots(2,1,figsize=(12,8))
    ax[0].plot(range(len(pred)), pred, c='r', label='Predicted data')
    ax[0].plot(range(len(pred)), y_test, c='b', label='Real data')
    ax[0].set_xlabel('Time')
    ax[0].set_ylabel('Number of passengers')
    ax[0].set_title(name)
    
    r2_scores = [r2_score(pred[:i], y_test[:i]) for i in range(3, 25)]
    ax[1].plot(range(3,25), r2_scores)
    ax[1].set_xlabel('Month of forecasting')
    ax[1].set_ylabel('$R^2$ score')

# Function that extracts data variables such as month and year from the date
def change_df(df, num_lag=12, time=True, lag=True):
    names = []
    if time:
        df['month'] = df.period.dt.month
        df['year'] = df.period.dt.year
        names += ['month', 'year'] 
    if lag:
        name_lags = []
        for i in range(1, num_lag):
            name = 'lag_' + str(i)
            name_lags.append(name)
            df[name] = df.number.shift(i)
        names += name_lags
    x = df[names]
    y = df['number']
    return x, y