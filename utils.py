import abc
import numpy as np
from  scipy import optimize
import pandas as pd
from time import time
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


from TS_Extrinsic_Regression.models.rocket import generate_kernels,apply_kernels

def sliding_windows(data, seq_length):
    x = []
    y = []

    for i in range(len(data)-seq_length-1):
        _x = data[i:(i+seq_length)]
        _y = data[i+seq_length]
        x.append(_x)
        y.append(_y)

    return np.array(x),np.array(y)

class LSTM(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()
        
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))
        
        c_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))
        
        # Propagate input through LSTM
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))
        
        h_out = h_out.view(-1, self.hidden_size)
        
        out = self.fc(h_out)
        
        return out


class MA():
    def __init__(self, q):
        self.q = q
        self.b = np.zeros(shape=(q,))

    def calculate_noise(self, sequence, b=None):
        if b is None:
            b = self.b

        q = b.shape[0]
        N = sequence.shape[0]

        seq_mean = np.mean(sequence)
        sequence = sequence - seq_mean
        noise = np.zeros(shape=(N,))

        for i in range(N):
            subnoise = np.flip(noise[max(0, i - q):i])

            if subnoise.shape[0] != q:
                subb = b[0:subnoise.shape[0]]
            else:
                subb = b
            
            noise[i] = sequence[i] - np.sum(subb * subnoise)

        return noise

    def predict(self, sequence, b=None, noise=None, steps=1):
        if b is None:
            b = self.b

        N = sequence.shape[0]
        q = b.shape[0]

        assert q <= N

        seq_mean = np.mean(sequence)
        sequence = sequence - seq_mean
        if noise is None:
            noise = self.calculate_noise(sequence, b)

        prediction = np.zeros(shape=(steps,))
        for i in range(steps):
            subnoise = np.flip(noise[N - q + i:N])
            subb = b[0:q - i]
            prediction[i] = np.sum(subb * subnoise) + seq_mean

        return prediction

    def loss_fn(self, b, sequence):
        q = b.shape[0]
        N = sequence.shape[0]
        noise = self.calculate_noise(sequence, b)
        loss = 0
        for i in range(q, N):
            pred = self.predict(sequence[:i], b, noise=noise[:i], steps=1)
            loss += (pred - sequence[i]) ** 2
        return loss

    def fit(self, sequence, method="BFGS"):
        b = optimize.minimize(self.loss_fn, np.zeros((self.q,)), args=(sequence,), method=method)
        self.b = b["x"]
        return b["x"]

def calculate_R2(model, test_sequence, start_with=20, step=1):
    #First start_with_steps will be used to forecast.
    loss = 0
    seq_mean = np.mean(test_sequence[start_with:])
    variance = 0
    for i in range(start_with, len(test_sequence) - step + 1):
        subsequence = test_sequence[:i]
        prediction = model.predict(subsequence, steps=step)[-1]
        loss += (test_sequence[i] - prediction) ** 2
        variance += (test_sequence[i] - seq_mean) ** 2
    return 1 - loss / variance

def calculate_R2_adjusted(model, test_sequence, start_with=20, step=1):
    R_2 = calculate_R2(model, test_sequence, start_with, step)
    N = len(test_sequence) - start_with 
    adjusted_R2 = 1 - (1 - R_2) * (N - 1) / (N - model.q - 1)
    return adjusted_R2
def split_train_test(y, train=90):
    y_train = y[:train]
    y_test = y[train:]
    return y_train, y_test

def find_best_q_for_ma(min_q,max_q,y_train,y_test):
    best_q = 1
    R2_ad_best = 0
    R2_1_history = []
    R2_2_history = []
    R2_adjected_history = []
    times = []
    for q in range(min_q, max_q):
        ma_model = MA(q=q)
        in_time = time()
        ma_model.fit(y_train)
        out_time = time()
        times.append(out_time - in_time)
        R2_1 = calculate_R2(ma_model, y_test, step=1)
        R2_2 = calculate_R2(ma_model, y_test, step=2)
        R2_ad = calculate_R2_adjusted(ma_model, y_test, step=1)
        R2_1_history.append(R2_1)
        R2_2_history.append(R2_2)
        R2_adjected_history.append(R2_ad)
    
    max_r2=max(R2_adjected_history)
    best_q1=np.where(np.array(R2_adjected_history)==max_r2)[0].item()
        
    plt.figure(figsize=(12, 8))
    plt.plot(R2_1_history,label='$R^2$ for 1-step ahead forecast ')
    plt.plot(R2_2_history,label='$R^2$ for 2-step ahead forecast')
    plt.plot(R2_adjected_history,label='$R^2$-adjusted for 1-step ahead forecast')
    plt.xlabel('$q$',fontsize=15)
    plt.ylabel('$R^2$',fontsize=15)
    plt.title('Choice of best MA parameter')
    plt.axvline(x=best_q1,ls='--',c='k',label=f"Best q parameter value for MA(q) = {best_q1}")
    plt.legend(loc=4)


    plt.show()
    return best_q1,times



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
def prediction(df, pred, test_n, model):
    columns = len(df.columns)
    pred = pred.to_numpy()
    pred = pred[:test_n]
    df[df.index>test_n] = 0
    for i in range(test_n-1, len(df)):
        pred = np.append(pred, model.predict(df[df.index==i]))
        if i < len(df):
            df[df.index==i+1] = pred[-columns-1:-1][::-1]
    return df, pred

def fit_predict(y,df_train, df_test,cv, num_lags=2, time=True, lag=True, name='Something'):
    df_full = pd.concat((df_train, df_test))
    x_full, y_full = change_df(df_full, num_lags, time, lag)
    x_test, y_test = x_full[x_full.index >= len(df_train)], y_full[y_full.index >= len(df_train)]
    # Extraction of date variables for train set
    X, Y = change_df(y, num_lags, time, lag)
    # Creation of grid to find the best params
    grid = {'n_estimators':[100,500,1000,2000], 'max_depth':[1,2,3,5], 
            'max_leaves':[1,2,5,10], 'n_jobs':[-1]}
    #grid={'max_depth':[1,2,3,5]}
    
    # Creation of GridSearchCV with custom cross validation folders
    # Folders were visualized earlier
    
    model = GridSearchCV(XGBRegressor(), param_grid=grid, cv=cv, verbose=1, scoring='r2')
    
    # Train of the model on train dataset
    model.fit(X, Y)
    
    # Predicting new values step by step:
    #x_test, y_test = change_df(y_validate, num_lags, time, lag)
    if lag: 
        temp, pred = prediction(x_full, y_full, 90, model)
        pred = pred[91:]
    else:
        pred = model.predict(x_test)
    
    fig,ax = plt.subplots(2,1,figsize=(8,6))
    ax[0].plot(range(len(pred)), pred, c='r', label='Predicted data')
    ax[0].plot(range(len(pred)), y_test, c='b', label='Real data')
    ax[0].set_xlabel('Time')
    ax[0].set_ylabel('Number of passengers')
    ax[0].set_title(name)
    ax[0].legend()
    
    r2_scores = [r2_score(pred[:i], y_test[:i]) for i in range(3, 25)]
    ax[1].plot(range(3,25), r2_scores)
    ax[1].set_xlabel('Month of forecasting')
    ax[1].set_ylabel('$R^2$ score')
    return model, pred