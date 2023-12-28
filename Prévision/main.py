# %%
import time,json,ccxt,math

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout
from os import getcwd,system
import matplotlib.pyplot as plt


from get_data import get_historical_from_db

# %%
def lire(chemin):
    with open (chemin,"r") as fichier:
        return json.load(fichier)

def enregistrer (chemin,contenu,modeLecture="w"):
    with open(chemin,modeLecture) as fichier :
        json.dump(contenu,fichier,indent=4, ensure_ascii=False, sort_keys=False )

def logs(message,chemin=str(getcwd())+"/logs.txt",affi=True):
    with open(chemin,"a") as l :
        l.write(time.ctime()+" : "+message+"\n")

# %%
paramètres=lire("/home/Astra_world/Astra_investissement/paramètre.json")
tf,chemin2="1h",""
chemin=getcwd()

with open(paramètres["chemin_liste_pair"],"r") as p:
    pair=p.readlines()[0]

df = get_historical_from_db(
        ccxt.binance(), 
        pair,
        tf,
        path=paramètres["chemin_binance"]
    )

# %%
regressor =load_model("/home/Astra_world/Astra_investissement/Entrainement/I.A loss")

# %%
data=df[:"2021"].filter(["close"])
dataset=data.values
training_data_len=math.ceil(len(dataset))


scaler=MinMaxScaler(feature_range=(0,1))
scaled_data=scaler.fit_transform(dataset)
data=df["2022"].filter(["close"])
dataset=data.values
training_data_len=math.ceil(len(dataset))
learn_period = 7

scaled_data=scaler.transform(dataset)


train_data=scaled_data[0:training_data_len,:]
x_train,y_train=[],[]
for i in range (learn_period,len(train_data)):
    x_train.append(train_data[i-learn_period:i,0])
    y_train.append(train_data[i,0])

x_train,y_train=np.array(x_train,dtype=object).astype(np.float32),np.array(y_train,dtype=object).astype(np.float32)



x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))



predictions=regressor.predict(x_train)
#print(predictions)
predictions=scaler.inverse_transform(predictions)

score = regressor.evaluate(x_train, predictions, verbose=2)
print(score)
predi=predictions
y_pred=data.values.tolist()[7:]
# Prédictions sur l'ensemble de test
y_test=predictions
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score


# Calcul de l'erreur absolue moyenne (MAE)
mae = mean_absolute_error(y_test, y_pred)

# Calcul de l'erreur quadratique moyenne (RMSE)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Calcul de l'erreur moyenne absolue en pourcentage (MAPE)
mape = mean_absolute_percentage_error(y_test, y_pred)

# Calcul du coefficient de corrélation (r)
r = np.corrcoef(y_test, y_pred)[0, 1]

# Calcul du coefficient de détermination (R²)
r2 = r2_score(y_test, y_pred)

# Calcul de l'erreur moyenne en pourcentage (MPE)
mpe = np.mean((y_test - y_pred) / y_test) * 100

# Affichage des résultats
print('MAE :', mae)
print('RMSE :', rmse)
print('MAPE :', mape)
print('r :', r)
print('R² :', r2)
print('MPE :', mpe)