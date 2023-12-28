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

df = get_historical_from_db("/home/Astra_world/MATIC-USDT.csv"
    )

# %%
regressor =load_model("/home/Astra_world/Astra_investissement/Entrainement/I.A")

# %%
scaler=MinMaxScaler(feature_range=(0,1))

df["scaler"]=scaler.fit_transform(df.filter(["close"]).values)
learn_period = 7

train_data=df["scaler"]
print(train_data[0:7])
x_train,y_train=[],[]
for i in range (learn_period,len(train_data)):
    x_train.append(train_data[i-learn_period:i])
    y_train.append(train_data[i,0])

x_train,y_train=np.array(x_train,dtype=object).astype(np.float32),np.array(y_train,dtype=object).astype(np.float32)


x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))


predictions=regressor.predict(x_train)
predictions = np.reshape(predictions, (predictions.shape[0], predictions.shape[1]))
predictions = scaler.inverse_transform(predictions)
df["predi"]=predictions
print(df)

"""score = regressor.evaluate(x_train, predictions, verbose=0)
print(score)
predi=predictions
reel=data.values.tolist()

y_pred_binary,c=0,0
# créer une liste de prédictions binaires
try:
    for i in range (1,len(predictions)):
        if predictions[i] > reel[i-1] and reel[i]>reel[i-1] or predictions[i] < reel[i-1] and reel[i]<reel[i-1]:
            y_pred_binary+=1
            c+=1
        else:
            #print(predictions[i],reel[i-1],reel[i])
            y_pred_binary-=1
except IndexError:
    pass

print('Accuracy:', y_pred_binary)
from sklearn.metrics import mean_squared_error
print(np.sqrt(mean_squared_error(reel[7:], predi)))"""

# %%
dollard,btc=100,0
max,min=0,100
history,Time=[],[]
salaire=2000        #Le salaire est retiré tout les mois si possible
pourcentage,compteur_heure=[],0
ancien=100
c2=0                #Compte le nombre de mouvement
dico={}
lc,historys=[],[]
del reel[0:7]
print(reel[:5],len(predi))
moins_heure=len(predi)-1
if not btc:
    btc=(dollard/reel[len(predi)-moins_heure-1][0])
    for i in range (len(predi)-moins_heure,len(predi)):
        historys.append(btc*reel[i-1][0])
else:
        historys.append(btc*reel[i-1][0])
dollard,btc=100,0

# %%

for i in range (len(predi)-moins_heure,len(predi)):
    compteur_heure+=1
    tt=dollard+btc*reel[i][0]
    logs(str(predi[i][0])+" "+str(reel[i][0]),str(getcwd()) + "prix.txt",False)
    if predi[i]-(reel[i][0]*1)>0 and dollard :
        logs("+ "+str(dollard)+" ----> "+str((dollard/reel[i][0])*0.999),str(getcwd()) + "prix.txt",False)
        btc=(dollard/reel[i][0])
        dollard=0
        c2+=1
        plt.scatter(i-len(predi), btc*reel[i][0], color='red', s=50, marker='+')
        print(time.asctime(time.gmtime(int(time.time())-(moins_heure-(i-len(predi)))*3600)),"+")
        #print(time.time(),(moins_heure,i,len(predi)))
    elif predi[i]-(reel[i][0]/1)<0 and btc:
        logs("- "+str(btc)+" ----> "+str((btc*reel[i][0])*0.999),str(getcwd()) + "prix.txt",False)
        dollard+=(btc*reel[i][0])
        btc=0
        c2+=1
        plt.scatter(i-len(predi), dollard, color='blue', s=50, marker='x')
        print(time.asctime(time.gmtime(int(time.time())-(moins_heure-(i-len(predi)))*3600)),"-")
        #print(time.time(),(moins_heure,i,len(predi)))
    #-------------------Enregistre les min et max------------------------------
    if tt<min:
        min=tt
    if tt>max:
        max=tt
    """#-------------------Gestion du salaire-------------------------------------
    if compteur_heure==720:
        if dollard-2000>2000:
            dollard-=2000
            #print(True,dollard)
        elif (btc*reel[i][0])-2000>2000:
            btc-=(2000/reel[i][0])
            #print(True,btc*reel[i][0])
        else:
            #print(False)
            pass
    #------------------Calcule les pourcentages obtenus par mois---------------
    if compteur_heure==720:
        lc.append(c2)
        compteur_heure,c2=0,0
        if dollard:
            pourcentage.append(dollard/ancien)
            ancien=dollard
        elif btc:
            pourcentage.append(((btc*reel[i][0])/ancien))
            ancien=btc*reel[i][0]"""
    #-----------------Ajoute les données dans le graphique----------------------
    history.append(dollard+btc*reel[i][0])
    Time.append(i-len(predi))
plt.title(pair+tf)
plt.plot(Time, history)
plt.plot(Time, historys)
plt.xlabel('Temps')
plt.ylabel('Wallet')
plt.show()
print(c2)

# %%
print(100*reel[-1][0]/reel[len(predi)-moins_heure][0])
print(history[-1],max,min)
del tt
for i in pourcentage:
    try:
        tt*=i
        #print(tt)
    except NameError:
        tt=i
try:
    print(tt)
except NameError:
    pass
for i in lc:
    print(i)


