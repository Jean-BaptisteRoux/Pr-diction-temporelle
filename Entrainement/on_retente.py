# %%
import math
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential,load_model
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
from get_data import get_historical_from_db
import ccxt,os
from sklearn.metrics import accuracy_score
plt.style.use("fivethirtyeight")

# %%
df = get_historical_from_db(
    ccxt.binance(), 
    "MATIC/USDT",
    "1h",
    path="/home/Astra_world/Data/"
)

# %%
data=df.filter(["close"])
dataset=data.values
training_data_len=math.ceil(len(dataset))

# %%
scaler=MinMaxScaler(feature_range=(0,1))
scaled_data=scaler.fit_transform(dataset)

# %%
train_data=scaled_data[0:training_data_len,:]
x_train,y_train=[],[]
for i in range (7,len(train_data)):
    x_train.append(train_data[i-7:i,0])
    y_train.append(train_data[i,0])

# %%
x_train,y_train=np.array(x_train,dtype=object).astype(np.float32),np.array(y_train,dtype=object).astype(np.float32)

# %%
x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))

# %%


# %%
try:
    regressor=load_model("/home/Astra_world/Astra_investissement/Entrainement/I.A")
except:
    regressor = Sequential()
    regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (x_train.shape[1], 1)))
    regressor.add(LSTM(units = 50, return_sequences = False))
    regressor.add(Dense(25))
    regressor.add(Dense(1))
    regressor.compile(optimizer="adam",loss="mean_squared_error")
    regressor.fit(x_train,y_train,batch_size=100,epochs=1)
    try:
        os.mkdir("I.A")
        os.mkdir("I.A.R")
    except FileExistsError:
        pass

# %%
data=df["2022"].filter(["close"])
#print(data)
dataset=data.values
training_data_len=math.ceil(len(dataset))

# %%
scaled_data=scaler.transform(dataset)

# %%
train_data=scaled_data[0:training_data_len,:]
X_train,Y_train=[],[]
for i in range (7,len(train_data)):
    X_train.append(train_data[i-7:i,0])
    Y_train.append(train_data[i,0])

# %%
X_train,Y_train=np.array(X_train,dtype=object).astype(np.float32),np.array(Y_train,dtype=object).astype(np.float32)

# %%
X_train=np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))

# %%
predictions=regressor.predict(X_train)
#print(predictions)
predictions=scaler.inverse_transform(predictions)

# %%
# Convertissez les prix en directions (1 pour une hausse, 0 pour une baisse)
y_pred = predictions
y_true = data.values[7:]

# Transformer les prédictions en une variable binaire (1 pour hausse, 0 pour baisse/stagnation)
y_pred_bin = np.where(y_pred > 0, 1, 0)
y_true_bin = np.where(y_true > 0, 1, 0)
print(y_true_bin)
# Calculer l'exactitude
accuracy = accuracy_score(y_true_bin, y_pred_bin)
print("Précision des prévisions directionnelles : {:.2%}".format(accuracy))
rmse=[accuracy_score(y_true_bin, y_pred_bin)]
print(rmse[-1])
diff=[100]
input("Entrainement ?")
while True:
    regressor.fit(x_train,y_train,batch_size=100,epochs=1)

    # %%
    data=df["2022":].filter(["close"])
    #print(data)
    dataset=data.values
    training_data_len=math.ceil(len(dataset))

    # %%
    scaled_data=scaler.transform(dataset)

    # %%
    train_data=scaled_data[0:training_data_len,:]
    X_train,Y_train=[],[]
    for i in range (7,len(train_data)):
        X_train.append(train_data[i-7:i,0])
        Y_train.append(train_data[i,0])

    # %%
    X_train,Y_train=np.array(X_train,dtype=object).astype(np.float32),np.array(Y_train,dtype=object).astype(np.float32)

    # %%
    X_train=np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))

    # %%
    predictions=regressor.predict(X_train)
    #print(predictions)
    predictions=scaler.inverse_transform(predictions)

    # %%
    
    rmse.append(regressor.evaluate(X_train, predictions, verbose=0))
    print(rmse[-2],rmse[-1])
    
    predi=predictions
    #print(len(predi))
    reel=data.values.tolist()
    #print(len(reel))

    # %%
    dollard,btc=100,0
    max,min=0,100
    history,Time=[],[]
    lc,historys=[],[]
    if not btc:
        btc=(dollard/reel[len(predi)-1][0])
        for i in range (len(predi)):
            historys.append(btc*reel[i-1][0])
    else:
        for i in range (len(predi)):
            historys.append(btc*reel[i-1][0])
    dollard,btc=100,0

    # %%
    c=0
    for i in range (len(predi)):
        tt=dollard+btc*reel[i-1][0]
        if predi[i]-reel[i-1]>0 and dollard :
            #print("monte",dollard,reel[i-1][0],predi[i],i-len(predi))
            btc=(dollard/reel[i-1][0])*0.999
            dollard=0
            c+=1
        elif predi[i]-reel[i-1]<0 and btc:
            dollard+=(btc*reel[i-1][0])*0.999
            #print("descend",dollard,reel[i-1][0],predi[i],i-len(predi))
            btc=0
            c+=1
        #-------------------Enregistre les min et max------------------------------
        if tt<min:
            min=tt
        if tt>max:
            max=tt
        #-----------------Ajoute les données dans le graphique----------------------
        history.append(dollard+btc*reel[i-1][0])
        Time.append(i-len(predi))
    print("Compteur :",c)

    # %%
    print(historys[-1])
    print(history[-1],max,min)
    print("Différence :"+str(history[-1]-historys[-1]))
    diff.append(history[-1]-historys[-1])
    if diff[-2]==diff[-1]:
        with open("/home/Astra_world/Astra_investissement/Entrainement/rendement.txt","a") as r:
            r.write(diff[-1])
        regressor.save("/home/Astra_world/Astra_investissement/Entrainement/I.A.R")
        print("save r")
    else:
        print("non")
        del diff[-1]
    if rmse[-2]>rmse[-1]:
        regressor.save("/home/Astra_world/Astra_investissement/Entrainement/I.A")
        print("save")
    else:
        print("non")
        del rmse[-1]
