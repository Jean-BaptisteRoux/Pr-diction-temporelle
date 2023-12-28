# %%
import time,json,math

import pandas as pd
import ccxt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout
from os import getcwd,system
import matplotlib.pyplot as plt


from get_data import get_historical_from_db



def lire(chemin):
    with open (chemin,"r") as fichier:
        return json.load(fichier)

def enregistrer (chemin,contenu,modeLecture="w"):
    with open(chemin,modeLecture) as fichier :
        json.dump(contenu,fichier,indent=4, ensure_ascii=False, sort_keys=False )

def logs(message,affi=True):
    with open(str(getcwd())+"/logs.txt","a") as l :
        l.write(time.ctime()+" : "+message+"\n")
    if affi:
        print(str(time.gmtime()[3])+":"+str(time.gmtime()[4])+":"+str(time.gmtime()[5])+" "+str(message))
# %%
while True:
    logs("Début du programme")
    pair,tf,chemin2="MATIC/USDT","1h",""
    chemin=getcwd()

    with open(getcwd()+"/pair.txt","w") as p:
        p.write(pair)
    
    """logs("Données en cours de téléchargement")
    retour=1
    while retour!=0:
        retour=system("python3 import\ données.py")
        if retour!=0:
            logs("Erreur de téléchargement de données")
            time.sleep(1)
            logs("Nouvelle tentative en cours")

    logs("Chargement des données")
    df = get_historical_from_db(
            ccxt.binance(), 
            pair,
            tf,
            path="/".join(getcwd().split("/")[:-1])+"/"
        )
    print(df)
    logs("Données chargées")
    logs("Sélection temporelle des données")
    training_set = df.copy().loc[:"2020"]
    test_set = df.copy().loc["2022":]
    learn_period = 7

    logs("Sélection temporelle terminée")

    # %%
    logs("Préparation de données")"""
    df = pd.read_csv("/home/Astra_world/Data/Binance/1h/MATIC-USDT copy.csv")
    df['date'] = pd.to_datetime(df['date'], unit='ms')
    df=df.set_index('date')
    dataset=df.filter(["close"]).values
    training_data_len=math.ceil(len(dataset))
    print(df["2023"])
    # Normalisation des données
    sc=MinMaxScaler(feature_range=(0,1))
    sc.fit_transform(dataset)
    """df = pd.read_csv("/home/Astra_world/Data/Binance/1h/MATIC-USDT.csv")
    df['date'] = pd.to_datetime(df['date'], unit='ms')
    df=df.set_index('date')
    dataset=df.filter(["close"]).values
    training_data_len=math.ceil(len(dataset))"""
    def données(scaler,dataset):
        scaled_data=scaler.transform(dataset)
        # Préparation des données pour l'entraînement
        learn_period = 7
        train_data=scaled_data[0:training_data_len,:]
        x_train,y_train=[],[]
        for i in range (learn_period,len(train_data)):
            x_train.append(train_data[i-learn_period:i,0])
            y_train.append(train_data[i,0])
        x_train,y_train=np.array(x_train,dtype=np.float32),np.array(y_train,dtype=np.float32)
        x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
        return x_train, y_train

    X_test, real_stock_price = données(sc,df["2023":].filter(["close"]).values[7:])
    logs("Préparation de données terminées")

    # %%
    logs("Chargement de l'I.A.")
    regressor =load_model("/home/Astra_world/Astra_investissement/Entrainement/I.A dollard gru")
    logs("I.A. chargé")
    # %%
    """logs("Traitement des nouvelles données")
    real_stock_price = test_set.iloc[:, 1:2].values
    dataset_total = pd.concat((training_set['close'], test_set['close']), axis = 0)
    inputs = dataset_total[len(dataset_total) - len(test_set) - learn_period:].values
    inputs = inputs.reshape(-1,1)
    inputs = sc.fit_transform(inputs)
    X_test = []
    for i in range(learn_period, len(test_set) + learn_period):
        X_test.append(inputs[i-learn_period:i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))"""
    predicted_stock_price = regressor.predict(X_test)
    predicted_stock_price = sc.inverse_transform(predicted_stock_price)
    logs("Traitement terminé")
    predi=predicted_stock_price
    reel=sc.inverse_transform(real_stock_price.reshape(-1, 1))

    

            
    while True:
        dollard,btc,max,min,history,Time,pourcentage,c,ancien,c2,lc,historys=100,0,0,100,[],[],[],0,100,0,[],[]
        print(len(predi))
        moins_heure=int(input("Heures: "))
        i=len(predi)-moins_heure
        print(dollard,btc,reel[i],predi[i])
        for i in range (len(predi)-moins_heure+1,len(predi)):
            c+=1
            tt=dollard+btc*reel[i]
            if predi[i]-(reel[i]*1)>0 and dollard :
                #print("+")
                btc=(dollard/reel[i])*0.999
                dollard=0
                plt.scatter(i, btc*reel[i], color='red', s=50, marker='x')
                c2+=1
            elif predi[i]-(reel[i]/1)<0 and btc:
                #print("-")
                dollard+=(btc*reel[i])*0.999
                plt.scatter(i, dollard, color='blue', s=50, marker='x')
                btc=0
                c2+=1
            if tt<min:
                min=tt
            if tt>max:
                max=tt
            """if c==720:
                lc.append(c2)
                c,c2=0,0
                if dollard:
                    #print(dollard,i)
                    pourcentage.append(dollard/ancien)
                    #print(dollard,ancien)
                    ancien=dollard
                elif btc:
                    pourcentage.append(((btc*reel[i])/ancien))
                    #print(btc*reel[i],ancien)
                    ancien=btc*reel[i]
                dollard,btc=0,124.875
                plt.title(pair+tf)
                plt.plot(Time, history)
                plt.xlabel('Temps')
                plt.ylabel('Wallet')
                plt.show()
                Time,history=[],[]
                if dollard-2000>2000:
                    dollard-=2000
                    print(True,dollard)
                elif (btc*reel[i])-2000>2000:
                    btc-=(2000/reel[i])
                    print(True,btc*reel[i])
                else:
                    print(False)"""
            history.append(dollard+btc*reel[i])
            Time.append(i)
        
        i=len(predi)-moins_heure+1
        dollard=100/reel[i]
        #print(dollard)
        #print(dollard*reel[i],dollard,reel[i],100/reel[i],reel[i])
        for i in range (len(predi)-moins_heure+1,len(predi)):
            #print(dollard*reel[i],dollard,reel[i])
            historys.append(dollard*reel[i])
        del tt
        #print(pourcentage,lc)
        print(100*reel[-1]/reel[len(predi)-moins_heure])
        #print(reel[0],reel[-1])
        print(history[-1],max,min)
        for i in pourcentage:
            try:
                tt*=i
                #print(tt)
            except NameError:
                tt=i
        dico={}
        for i in lc:
            try:
                if dico[str(i)]:
                    dico[str(i)]+=1
            except KeyError:
                dico[str(i)]=1
        for i in dico.keys():
            print(i,dico[i]/len(predi)*100)
        plt.title(pair+tf)
        plt.plot(Time, history)
        plt.plot(Time, historys)
        plt.xlabel('Temps')
        plt.ylabel('Wallet')
        plt.show()
        break
    break