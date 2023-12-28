# %%
import time,json,math

import pandas as pd
import ccxt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from os import getcwd,system
import matplotlib.pyplot as plt
import datetime
import asyncio



def get_historical_from_db(exchange, symbol, timeframe, path="database/"):
    symbol = symbol.replace('/','-')
    df = pd.read_csv(filepath_or_buffer=path+"Data/"+str(exchange.name)+"/"+timeframe+"/"+symbol+".csv")
    df = df.set_index(df['date'])
    df.index = pd.to_datetime(df.index, unit='ms')
    del df['date']
    return df

def get_historical_from_path(path):
    df = pd.read_csv(filepath_or_buffer=path)
    df = df.set_index(df['date'])
    df.index = pd.to_datetime(df.index, unit='ms')
    del df['date']
    return df



def lire(chemin):
    with open (chemin,"r") as fichier:
        return json.load(fichier)

def enregistrer (chemin,contenu,modeLecture="w"):
    with open(chemin,modeLecture) as fichier :
        json.dump(contenu,fichier,indent=4, ensure_ascii=False, sort_keys=False )

def logs(message,niveau=0,affi=True):
    for i in range (niveau+1):
        with open(paramètres["chemin_logs"]+"logs"+str(i)+".txt","a") as l :
            l.write(time.ctime()+" : "+message+"\n")
    if affi:
        print(str(time.gmtime()[3])+":"+str(time.gmtime()[4])+":"+str(time.gmtime()[5])+" "+str(message))
# %%
paramètres=lire("/home/Astra_world/Astra_investissement/paramètre.json")
clés=lire("/home/Astra_world/Astra_investissement/keys.json")
with open(paramètres["chemin_liste_pair"],"r") as p:
    pair=p.readlines()
tf="1h"
logs("Chargement de l'I.A.")
regressor =load_model(paramètres["chemin_ia_maticusdt_1h"])
logs("I.A. chargé",1)

#149.75
while True:
    logs("Début du programme",1)
    # récupérer l'heure actuelle
    now = datetime.datetime.now()
    # attendre jusqu'à l'heure suivante
    logs("On patiente "+str((59-now.minute)*60 + (60-now.second))+" secondes")
    #time.sleep((59-now.minute)*60 + (60-now.second))
    # Récupération du portefeuille
    exchange=ccxt.binance()
    exchange.apiKey = clés["api_key"]
    exchange.secret = clés["api_secret"]
    balance = exchange.fetch_balance()
    wallet = balance['total']

    # Enregistrement du portefeuille dans un fichier CSV
    df_wallet = pd.DataFrame.from_dict(wallet, orient='index', columns=['balance'])
    df_wallet=df_wallet.T
    df_wallet=df_wallet.set_index(["test"])
    print(df_wallet)
    df_wallet.to_csv('portefeuille.csv', index_label='currency')
    break
    for boucle in pair:
        retour=1
        while retour!=0:
            retour=system("python3 "+paramètres["chemin_import_données"])
            if retour!=0:
                logs("Erreur de téléchargement de données")
                time.sleep(1)
                logs("Nouvelle tentative en cours")

        logs("Chargement des données")
        df = get_historical_from_db(
                ccxt.binance(), 
                boucle,
                tf,
                path=paramètres["chemin_binance"]
            )
        logs("Données chargées",1)
        data=df[:"2020"].filter(["close"])
        dataset=data.values
        training_data_len=math.ceil(len(dataset))


        # %%
        scaler=MinMaxScaler(feature_range=(0,1))
        scaled_data=scaler.fit_transform(dataset)
        logs("Données chargées",1)
        logs("Sélection temporelle des données")
        data=df["2022":].filter(["close"])
        dataset=data.values
        training_data_len=math.ceil(len(dataset))
        learn_period = 7

        logs("Sélection temporelle terminée")

        # %%
        logs("Préparation de données")
        scaled_data=scaler.transform(dataset)


        train_data=scaled_data[0:training_data_len,:]
        x_train,y_train=[],[]
        for i in range (learn_period,len(train_data)):
            x_train.append(train_data[i-learn_period:i,0])
            y_train.append(train_data[i,0])

        x_train,y_train=np.array(x_train,dtype=object).astype(np.float32),np.array(y_train,dtype=object).astype(np.float32)
        


        x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
        logs("Préparation de données terminées",1)


        # %%
        logs("Traitement des nouvelles données")
        # %%
        predictions=regressor.predict(x_train)
        #print(predictions)
        predictions=scaler.inverse_transform(predictions)

        # %%
        rmse=[(np.sqrt(np.mean(predictions-y_train)**2))]
        logs("Traitement terminé",1)
        predi=predictions
        reel=data.values.tolist()

        with open(paramètres["chemin_predi"],"w") as p:
            p.write(str(predi[-1])[1:-1])
        with open(paramètres["chemin_reel"],"w") as p:
            p.write(str(reel[-2])[1:-1])
        with open(paramètres["chemin_pair"],"w") as p:
            p.write(str(boucle))
        retour=1
        while retour!=0:
            retour=system("python3 "+paramètres["chemin_choix"])
            if retour!=0:
                logs("Erreur de connexion")
                time.sleep(1)
                logs("Nouvelle tentative en cours")

