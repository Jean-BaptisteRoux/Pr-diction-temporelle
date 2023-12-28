# %%

import json,os

import pandas as pd
import ccxt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout
from os import getcwd
import matplotlib.pyplot as plt

from get_data import get_historical_from_db



def lire(chemin):
    with open (chemin,"r") as fichier:
        return json.load(fichier)

def enregistrer (chemin,contenu,modeLecture="w"):
    with open(chemin,modeLecture) as fichier :
        json.dump(contenu,fichier,indent=4, ensure_ascii=False, sort_keys=False )

#, "BTC/USDT", "ETH/USDT", 'ADA/USDT', 'XRP/USDT', 'BNB/USDT', 'LINK/USDT', 'LTC/USDT', "DOGE/USDT", "SOL/USDT", "AVAX/USDT", "DOT/USDT", "LUNA/USDT", "NEAR/USDT", "EGLD/USDT", "XTZ/USDT", "AAVE/USDT", "UNI/USDT", "FTM/USDT", "BCH/USDT"
# %%
for pair in [ "MATIC/USDT"]:
    for tf in ["1h"]:
        print(os.getcwd()+"/stat.txt")
        with open(os.getcwd()+"/statistique"+str(pair.split("/")[0])+str(pair.split("/")[1])+str(tf)+".txt","w") as s:
            s.write("")
        for itéra in range (100):
            for niveau in range (10):
                for nereunos in range (100):
                    with open (os.getcwd()+"/stat.txt","r") as s:
                        stat=s.readlines()
                    stat1=[]
                    for topr in stat :
                        stat1.append(topr[:len(pair)+len(tf)+len(str(itéra)+"N"+str(niveau)+"n"+str(nereunos))])
                    if pair+tf+str(itéra)+"N"+str(niveau)+"n"+str(nereunos)  not in  stat1:

                        df = get_historical_from_db(
                            ccxt.binance(), 
                            pair,
                            tf,
                            path="/home/Astra_world/"
                        )
                        """print(df)
                        print(df[:"2020"]["close"])"""
                        training_set = df.copy().loc[:"2020"]
                        test_set = df.copy().loc["2022"]
                        learn_period = 7

                        # %%
                        #print(df[:"2020"]["close"])
                        sc = MinMaxScaler(feature_range = (0, 1))
                        """print(df[:"2020"])
                        print(training_set["close"])
                        print(df["2022"])
                        print(test_set["close"])"""
                        training_set_scaled = sc.fit_transform(training_set["close"].values.reshape(-1, 1))
                        

                        X_train = []
                        y_train = []

                        for i in range(learn_period, len(training_set)):
                            X_train.append(training_set_scaled[i-learn_period:i, 0])
                            y_train.append(training_set_scaled[i, 0])
                        
                        X_train, y_train = np.array(X_train), np.array(y_train)


                        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
                        #print(X_train[0][1:2])

                    # %%

                        if True:
                            #print(nereunos)
                            regressor = Sequential()

                            regressor.add(LSTM(units = 500, return_sequences = True, input_shape = (X_train.shape[1], 1)))
                            regressor.add(Dropout(0.2))

                            regressor.add(LSTM(units = 500, return_sequences = True))
                            regressor.add(Dropout(0.2))

                            regressor.add(LSTM(units = 500, return_sequences = True))
                            regressor.add(Dropout(0.2))

                            regressor.add(LSTM(units = 500, return_sequences = True))
                            regressor.add(Dropout(0.2))

                            regressor.add(LSTM(units = 500, return_sequences = True))
                            regressor.add(Dropout(0.2))

                            regressor.add(LSTM(units = 500, return_sequences = True))
                            regressor.add(Dropout(0.2))

                            regressor.add(LSTM(units = 500, return_sequences = True))
                            regressor.add(Dropout(0.2))

                            regressor.add(LSTM(units = 500))
                            regressor.add(Dropout(0.2))

                            regressor.add(Dense(units = 1))

                            regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

                            regressor.fit(X_train, y_train, epochs = 50, batch_size = 32)
                        else:
                            regressor =load_model("/home/Astra_world/Astra_investissement/I_A_/MATIC/USDT/1h")
                        # %%
                        real_stock_price = test_set.iloc[:, 3:4].values
                        """print(test_set.iloc)
                        print(test_set.iloc[:, 3:4])
                        print(test_set.iloc[:, 1:2].values)
                        print(test_set.iloc[:].values)"""
                        dataset_total = pd.concat((training_set['close'], test_set['close']), axis = 0)
                        inputs = dataset_total[len(dataset_total) - len(test_set) - learn_period:].values
                        inputs = inputs.reshape(-1,1)
                        inputs = sc.fit_transform(inputs)
                        X_test = []
                        for i in range(learn_period, len(test_set) + learn_period):
                            X_test.append(inputs[i-learn_period:i, 0])
                        X_test = np.array(X_test)
                        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
                        predicted_stock_price = regressor.predict(X_test)
                        predicted_stock_price = sc.inverse_transform(predicted_stock_price)

                        predi=predicted_stock_price
                        reel=real_stock_price
                        dollard,btc,max,min=100,0,0,100
                        for i in range (1,len(predi)):
                            tt=dollard+btc*reel[i-1]
                            if predi[i]-reel[i-1]>0 and dollard :
                                btc=(dollard/reel[i-1][0])*0.999
                                dollard=0
                            elif predi[i]-reel[i-1]<0 and btc:
                                dollard+=(btc*reel[i-1][0])*0.999
                                btc=0
                            if tt<min:
                                min=tt
                            if tt>max:
                                max=tt
                            #print(dollard,btc)
                        #print(reel[-1][0])
                        print(dollard+btc*reel[-1][0],max,min)
                        break
                        with open (os.getcwd()+"/stat.txt","a") as s:
                            s.write(pair+tf+str(itéra)+"N"+str(niveau)+"n"+str(nereunos)+" // "+str(dollard+btc*reel[-1])+" // "+str(max)+" // "+str(min)+"\n")
                        try:
                            with open (os.getcwd()+"/statistique"+str(pair.split("/")[0])+str(pair.split("/")[1])+str(tf)+".txt","r") as s:
                                stat=s.readlines()
                                print(os.getcwd()+"/statistique"+str(pair.split("/")[0])+str(pair.split("/")[1])+str(tf)+".txt")
                            if float(stat[0])<dollard+btc*reel[-1]:
                                print(float(stat[0])<dollard+btc*reel[-1])
                                with open (os.getcwd()+"/statistique"+str(pair.split("/")[0])+str(pair.split("/")[1])+str(tf)+".txt","w") as s:
                                    s.write(str(dollard+btc*reel[-1])[1:-1])
                                pair1,pair2=pair.split("/")[0],pair.split("/")[1]
                                try:
                                    os.mkdir("I.A")
                                except FileExistsError:
                                    pass
                                try:
                                    os.mkdir("I.A/"+pair1)
                                except FileExistsError:
                                    pass
                                try:
                                    os.mkdir("I.A/"+pair1+"/"+pair2)
                                except FileExistsError:
                                    pass
                                try:
                                    os.mkdir("I.A/"+pair1+"/"+pair2+"/"+tf)
                                except FileExistsError:
                                    pass
                                print("Enresgistrer")
                                regressor.save(getcwd()+"/I.A/"+pair+"/"+tf)
                        except IndexError as e:
                            print(e)
                            if not stat:
                                with open (os.getcwd()+"/statistique"+str(pair.split("/")[0])+str(pair.split("/")[1])+str(tf)+".txt","w") as s:
                                    s.write(str(dollard+btc*reel[-1])[1:-1])
                            pair1,pair2=pair.split("/")[0],pair.split("/")[1]
                            try:
                                os.mkdir("I.A")
                            except FileExistsError:
                                pass
                            try:
                                os.mkdir("I.A/"+pair1)
                            except FileExistsError:
                                pass
                            try:
                                os.mkdir("I.A/"+pair1+"/"+pair2)
                            except FileExistsError:
                                pass
                            try:
                                os.mkdir("I.A/"+pair1+"/"+pair2+"/"+tf)
                            except FileExistsError:
                                pass
                            print("Enresgistrer")
                            regressor.save(getcwd()+"/I.A/"+pair+"/"+tf)