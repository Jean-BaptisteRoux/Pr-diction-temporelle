import numpy as np
import pandas as pd
#import logging as l
import os,time,sys
from shutil import rmtree,copytree
from math import ceil,isnan
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from time import time, gmtime, localtime,sleep,strftime
from tensorflow import keras, constant, reshape,float32,int64
from tensorflow.train import Feature,FloatList, Int64List,SequenceExample,FeatureLists
from tensorflow.keras import layers
from tensorflow.data import Dataset
from tensorflow.keras.constraints import MaxNorm
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from requests import get
from optuna import create_study,load_study,trial
#from optuna.trial.TrialState import FAILURE,USER_TERMINATED
from optuna.samplers import TPESampler
from optuna.exceptions import DuplicatedStudyError
from pickle import dump
from json import load,dump as d




def entrainement(model,X_train, valid,epochs,batch_size,crypto,interval,source):
    """La fonction entrainement prennant pour arguments l'I.A., les 4 listes d'entrainement, le nombre d'entrainement,
    le segmentage de l'entrainement, ainsi que l'affichage. Cette fonction permet d'entrainer une I.A."""
    
    for i in tqdm(range(epochs),desc=crypto+" "+interval):
        try:
            history=model.fit(X_train, batch_size=batch_size, epochs=1, validation_data=valid, verbose=0)
        except KeyboardInterrupt:
            autre=input("Voulez-vous arrêtez ?")
            if autre=="y" or autre=="yes":
                os._exit(0)
        if isnan(history.history['loss'][0]):
            print("Training stopped: NaN loss")
            return 0
    return model

def objective(trial,crypto,interval,source="Binance",parent_folder = os.path.dirname(os.path.abspath(__file__)),affiche=True):
    """Cette fonction clée centralise la création d'I.A.. Elle prends pour arguments l'essai, la crypto, l'intervalle
    de temps et le dossier source. parent_folder=chemin du fichier fonction.py"""
    # Définition des hyperparamètres à optimiser
    clipvalue = trial.suggest_float('clipvalue', 0.01, 1.0)
    dropout = trial.suggest_float("dropout", 0, 0.5)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1)
    num_units = trial.suggest_int("num_units", 1, 512)
    num_layers = trial.suggest_int("num_layers", 1, 5)
    bias_initializer = trial.suggest_categorical("bias_initializer", ["zeros", "ones", "random_normal", "random_uniform"])
    kernel_initializer = trial.suggest_categorical("kernel_initializer", ["glorot_uniform", "glorot_normal", "he_normal", "he_uniform"])
    activation = trial.suggest_categorical("activation", ["relu", "sigmoid", "tanh"])
    units = trial.suggest_int("units", 1, 10)
    epochs = trial.suggest_int("epochs", 1, 80)
    batch_size = trial.suggest_int("batch_size", 16, 128)
    recurrent_dropout = trial.suggest_float("recurrent_dropout", 0, 0.5)
    l1_lambda = trial.suggest_float("l1_lambda", 1e-5, 1e-1)
    l2_lambda = trial.suggest_float("l2_lambda", 1e-5, 1e-1)
    learn_period = trial.suggest_int("learn_period", 1,100)

    df=data(crypto,interval,source)
    X_train,valid,dr,r,scaler=données_séparées(df,learn_period,batch_size)
    reel=df.loc["2023":].filter(["close"])
    print(reel.iloc[-1].values , reel.iloc[0].values,reel.iloc[-1].values / reel.iloc[0].values)
    courbe=100*(reel.iloc[-1].values / reel.iloc[0].values)

    # Définition du modèle GRU
    model = keras.Sequential()
    for i in range(num_layers):
        model.add(layers.GRU(num_units,
                            dropout=dropout,
                            recurrent_dropout=recurrent_dropout,
                            bias_initializer=bias_initializer,
                            kernel_initializer=kernel_initializer,
                            activation=activation,
                            kernel_regularizer=keras.regularizers.L1L2(l1=l1_lambda, l2=l2_lambda),
                            return_sequences=True if i < num_layers - 1 else False,
                            input_shape=(X_train.element_spec[0].shape[1], X_train.element_spec[0].shape[2])))
    model.add(layers.Dense(units,activation=activation, kernel_constraint=MaxNorm(clipvalue)))
    model.add(layers.Flatten())
    model.add(layers.Dense(1,activation=activation, kernel_constraint=MaxNorm(clipvalue)))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                loss=keras.losses.MeanSquaredError(),
                metrics=[keras.metrics.RootMeanSquaredError()])
    print(f"dropout={dropout:.2f}, learning_rate={learning_rate:.5f}, "
        f"num_units={num_units}, num_layers={num_layers}, bias_initializer={bias_initializer}, "
        f"kernel_initializer={kernel_initializer}, activation={activation}, units={units}, "
        f"recurrent_dropout={recurrent_dropout:.2f}, epochs={epochs}, batch_size={batch_size}, "
        f"l1_lambda={l1_lambda:.5f}, l2_lambda={l2_lambda:.5f}")
    print("Learn_period : ",learn_period,"\n")
    start_time = time()
    model=entrainement(model,X_train, valid,epochs,batch_size,crypto,interval,source)
    if model==0:
        return 0
    """model.fit(X_train, batch_size=batch_size, epochs=epochs, validation_data=valid, verbose=1)"""

    end_time = time()
    # Affichage des résultats
    print(f"time={end_time-start_time:.2f}s")
    #Calcul l'efficacité du modèle
    résultat=simulation(scaler.inverse_transform(model.predict(dr).reshape(-1, 1)).reshape(-1),scaler.inverse_transform(r.reshape(-1, 1)).reshape(-1))
    #Enregistrement du model et de sa performance
    print("Résultat en net : "+str(résultat),courbe)
    chemin=os.path.join(parent_folder,source,crypto,interval)+"/"
    try:
        os.makedirs(chemin+str(trial.number))
    except FileExistsError:
        pass
    #model.save(os.path.join(os.path.dirname(os.path.abspath(sys.executable)), source,crypto,interval,trial.number))
    model.save(chemin+str(trial.number))
    """with open(chemin+str(trial.number)+"/scaler.pickle","wb") as best:
        dump(scaler.pickle,best)"""
    with open(chemin+crypto+"-"+interval+".txt","a") as best:
        best.write(str(résultat)+" "+str(résultat-courbe)+"\n") 
    # Retour de la valeur à optimiser (ici, le dollard)
    return float(résultat)

def simulation(predictions,reel,taux=1-0.999):
    """Cette fonction simule l'achat et le vente d'un actif selon la différence entre predictions[i] et reel[i]"""
    dollard,btc=100,0
    for i in range(1,len(predictions)): #On part de 1 pour éviter une erreur à i=0
        #print(predictions[i],reel[i])
        if predictions[i]>reel[i] and dollard:#On achète
            btc=(dollard/reel[i])*(1-taux)
            dollard=0
        elif predictions[i]<reel[i] and btc:#On vend
            dollard=(btc*reel[i])*(1-taux)
            btc=0
    dollard+=(btc*reel[-1])
    print(dollard,(100*reel[-1]/reel[0]),reel[-1],reel[0])
    """dollard-=(100*reel[-1]/reel[0])"""
    print("Résultat brut : "+str(dollard))
    return dollard

def données(scaler,dataset,learn_period):
    """Cette fonction permet de transformer une série"""
    scaled_data=scaler.transform(dataset)
    # Préparation des données pour l'entraînement
    training_data_len=ceil(len(dataset))
    train_data=scaled_data[0:training_data_len,:]
    x_train,y_train=[],[]
    for i in range (learn_period,len(train_data)):
        x_train.append(train_data[i-learn_period:i,0])
        y_train.append(train_data[i,0])
    x_train,y_train=np.array(x_train,dtype=np.float32),np.array(y_train,dtype=np.float32)
    x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
    return x_train, y_train

def create_dataset(inputs, outputs, batch_size=1):
    # Convertir les listes d'entrée et de sortie en tableaux numpy
    inputs_np = np.array(inputs)
    outputs_np = np.array(outputs)

    # Créer un objet tf.data.Dataset à partir des tableaux numpy
    dataset = Dataset.from_tensor_slices((inputs_np, outputs_np))

    # Mélanger les données
    dataset = dataset.shuffle(buffer_size=len(inputs))

    # Regrouper les données en lots
    if batch_size is not None:
        dataset = dataset.batch(batch_size)

    return dataset

def actualiser_csv(folder,folder2,end_time=1682113051788,start_time = 1390694400000,parent_folder = os.path.dirname(os.path.abspath(__file__))):
    # URL de l'API de Binance
    url = "https://api.binance.com/api/v3/klines"

    # Paramètres pour la requête
    limit = 1000  # nombre de bougies à récupérer à chaque requête
    try:
        df=pd.read_csv(os.path.join(parent_folder,"Binance", folder+"-USDT",folder2,folder+"-USDT.csv"))
        del df["date"][len(df["date"])-1]
        start_time = df["date"][len(df["date"])-1]+1
    except FileNotFoundError:
        start_time = 1390694400000 # timestamp de début (01/27/2014 à 00:00:00 UTC)
        df = pd.DataFrame()
    #end_time = 1681332000000  # timestamp de fin (03/03/2022 à 00:00:00 UTC)

    # Récupération des données pour chaque paire de cryptomonnaies  # Initialisation du dataframe pour la paire actuelle
    # Récupération des données pour chaque période de 1000 bougies

    while start_time < end_time:
        params = {"symbol": folder.replace("-",""), "interval": folder2, "startTime": start_time, "endTime": end_time, "limit": limit}
        response = get(url, params=params)
        json_data = response.json()
        # Extraction des données de chaque bougie
        data = [[candle[0], float(candle[1]), float(candle[2]), float(candle[3]), float(candle[4]), float(candle[5])] for candle in json_data]

        # Création du dataframe pour la période actuelle et ajout à celui de la paire
        df = pd.concat([df, pd.DataFrame(data, columns=["date", "open", "high", "low", "close", "volume"])], ignore_index=True)

        # Mise à jour du timestamp de début pour la prochaine période
        try:
            start_time = int(json_data[-1][0]) + 1
        except:
            break

    # Écriture des données dans un fichier CSV pour la paire actuelle
    df.to_csv(os.path.join(parent_folder,"Binance", folder,folder2,folder+".csv"), index=False)

def nouvelle_étude_binance (crypto,interval,end_time=1682113051788,start_time = 1390694400000,parent_folder = os.path.dirname(os.path.abspath(__file__))):
    folder=crypto
    folder2=interval

    try:
        os.mkdir(os.path.join(parent_folder,"Binance", folder+"-USDT"))
    except :
        pass
    try:
        os.mkdir(os.path.join(parent_folder,"Binance", folder+"-USDT",folder2))
    except:
        pass

    actualiser_csv(folder, folder2)
    print(os.path.join(parent_folder,"Binance", folder+"-USDT",folder2,folder+"-USDT.csv"))
    try:
        create_study(study_name=crypto+interval, storage="sqlite:///"+os.path.join(parent_folder,"Binance", crypto, interval, crypto+"-"+interval+".db"), direction="maximize")
    except DuplicatedStudyError:
        pass
    print("Étude créée !")

def data(crypto,interval,source="Binance"):
    parent_folder = os.path.dirname(os.path.abspath(__file__))
    chemin=os.path.join(parent_folder,source,crypto,interval,crypto+".csv")
    try:
        df = pd.read_csv(chemin)
        df['date'] = pd.to_datetime(df['date'], unit='ms')
        df=df.set_index('date')
    except FileNotFoundError as e:
        nouvelle_étude_binance(crypto,interval)
        df = pd.read_csv(chemin)
        df['date'] = pd.to_datetime(df['date'], unit='ms')
        df=df.set_index('date')
    return df

def données_séparées(df,learn_period,batch_size,année=localtime().tm_year):
    scaler=MinMaxScaler(feature_range=(0,1))
    scaler.fit_transform(df[:str(année-1)].filter(["close"]).values)
    dfy=df.iloc[:].drop(df[str(année):].index).filter(["close"])
    n=dfy.iloc[:int(dfy.shape[0] * 0.8)]
    m=dfy.iloc[:].drop(n.index)
    i=données(scaler,n.values,learn_period)
    u=données(scaler,m.values,learn_period)
    un=create_dataset(i[0],i[1],batch_size)
    trois=create_dataset(u[0],u[1],batch_size)
    deux=données(scaler,df[str(année):].filter(["close"]).values,learn_period)
    return un,trois,deux[0],deux[1],scaler

def lancement_étude(crypto,interval,limite,source="Binance",parent_folder = os.path.dirname(os.path.abspath(__file__))):
    print(crypto,interval)
    if not os.path.exists(os.path.join(parent_folder,source)):
        os.makedirs(os.path.join(parent_folder,source))
    if not os.path.exists(os.path.join(parent_folder,source, crypto)):
        os.makedirs(os.path.join(parent_folder,source, crypto))
    if not os.path.exists(os.path.join(parent_folder,source, crypto, interval)):
        os.makedirs(os.path.join(parent_folder,source, crypto, interval))
    try:
        study = load_study(study_name=crypto+interval, storage="sqlite:///"+os.path.join(parent_folder,source, crypto, interval, crypto+interval+".db"),sampler=TPESampler())
    except:
        print(crypto+interval,os.path.join(parent_folder,source, crypto, interval, crypto+interval+".db"))    
        study = create_study(study_name=crypto+interval, storage="sqlite:///"+os.path.join(parent_folder,source, crypto, interval, crypto+interval+".db"), direction="maximize")
    #study.pruner(lambda t: t.state == FAILURE or t.state == USER_TERMINATED)
    # Votre code ici
    #print(study.trials_dataframe())
    #study.trials_dataframe().to_csv(os.path.join(parent_folder,source,crypto,interval,crypto+interval+".csv"), index=False)
    # Récupérer la liste des essais échoués
    #study.clean_failed_trials()
    limite=int(limite)
    for i in range(limite):
        func = lambda trial: objective(trial, crypto,interval,source="Binance",parent_folder = os.path.dirname(os.path.abspath(__file__)),affiche=False)
        study.optimize(func, n_trials=1)
        print("\n\n\n\n\n---------------------------------\n\n\n\n\n")
        with open(parent_folder+"/"+source+".txt","r") as e:
            ordre=e.readlines()
        if limite>1:
            for o in range(len(ordre)):
                if ordre[o].split(" ")[0]==crypto and ordre[o].split(" ")[1]==interval:
                    ordre[o]=ordre[o][:-len(str(limite-i))-1]+str(limite-i)+"\n"
        else:
            del ordre[0]
        with open(parent_folder+"/"+source+".txt","w") as e:
            for j in ordre:
                e.write(str(j))
    study.trials_dataframe().to_csv(os.path.join(parent_folder,source,crypto,interval,crypto+interval+".csv"), index=False)

def lancement_des_études(source="Binance",interval_list = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w"],parent_folder = os.path.dirname(os.path.abspath(__file__))):
    """Lance l'entrainement d'une étude à partir du fichier source.txt"""
    """# Obtenez le chemin absolu du répertoire contenant le fichier exécutable
    app_dir = os.path.dirname(os.path.abspath(sys.executable))

    # Créez un sous-dossier nommé "data" dans le répertoire de l'application
    data_dir = os.path.join(app_dir, source)
    os.makedirs(data_dir, exist_ok=True)"""
    with open(parent_folder+"/"+source+".txt","r") as e:
        ordre=e.readlines()
    if len(ordre)==1 and ordre[0].split(" ")[0] in interval_list:
        try:
            nombre=ordre[0].split(" ")[1]
        except IndexError:
            nombre=input("Le nombre d'itération n'est pas indiqué. Combien de fois je m'entraine ?")
        except ValueError:
            nombre=input("Le nombre d'itération n'est pas indiqué. Combien de fois je m'entraine ?")
        with open(parent_folder+"/"+source+".txt","w") as e:
            for i in [f for f in os.listdir(parent_folder+"/"+source) if os.path.isdir(os.path.join(parent_folder,source, f))]:
                e.write(str(i)+" "+str(ordre[0].split(" ")[0])+" "+str(nombre)+"\n")
    elif len(ordre)==1 and ordre[0].split(" ")[0]=="*":
        try:
            nombre=int(ordre[0].split(" ")[1])
        except IndexError:
            nombre=input("Le nombre d'itération n'est pas indiqué. Combien de fois je m'entraine ?")
        except ValueError:
            nombre=input("Le nombre d'itération n'est pas indiqué. Combien de fois je m'entraine ?")
        with open(parent_folder+"/"+source+".txt","w") as e:
            for i in [f for f in os.listdir(parent_folder+"/"+source) if os.path.isdir(os.path.join(parent_folder,source, f))]:
                for j in interval_list:
                    e.write(i+" "+j+" "+str(nombre)+"\n")
    with open(parent_folder+"/"+source+".txt", "r+") as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines if line.strip()]
        lines.sort()
        f.seek(0)
        f.write('\n'.join(lines))
        f.truncate()

    while True:
        with open(parent_folder+"/"+source+".txt","r") as e:
            ordre=sorted(e.readlines())
        if not ordre:
            os._exit(0)
        # Diviser chaque ligne en trois parties en utilisant un espace comme séparateur
        data = []
        for line in ordre:
            if line:
                parts = line.strip().split(' ')
                data.append(parts)

        

        columns=['crypto', 'interval', 'itération']
        # Créer un DataFrame à partir de la liste avec trois colonnes
        df = pd.DataFrame(data, columns=columns)
        print(df)
        for index, row in df.iterrows():
            """# Créez un sous-dossier nommé "data" dans le répertoire de l'application
            data_dir = os.path.join(app_dir, source,row["crypto"],row["interval"])
            os.makedirs(data_dir, exist_ok=True)"""
            print(row["crypto"],row["interval"],row["itération"])
            if row["crypto"] and row["interval"] and row["itération"]:
                lancement_étude(row["crypto"],row["interval"],row["itération"],source=source)

def ajout_hyperparamètre(hyperparamètre: str,valeur: int,study_name:str,storage:str):
    """Cette fonction permet d'ajouter un hyperparamètre à une étude"""
    # Votre étude Optuna existante ici.
    study =load_study(study_name=study_name, storage=storage)

    # Ajouter un hyperparamètre avec une valeur fixe à tous les essais.
    for trial in study.trials:
        if hyperparamètre not in trial.user_attrs:
            trial.set_user_attr(hyperparamètre, valeur)

def actualisation_étude(crypto,interval,source="Binance",parent_folder = os.path.dirname(os.path.abspath(__file__))):
    print("sqlite:///"+os.path.join(parent_folder,source, crypto, interval, crypto+"-"+interval+".db"))
    study = load_study(study_name=crypto+"-"+interval, storage="sqlite:///"+os.path.join(parent_folder,source, crypto, interval, crypto+"-"+interval+".db"),sampler=TPESampler())
    df = study.trials_dataframe()
    for i,row in df:
        model=load_model(os.path.join(parent_folder,source,crypto,interval,row["number"]))
        X,y,dr,r=données_séparées(data(crypto,interval,source),row['learn_period'])
        trial = study.get_trial(row["number"])
        trial_params = trial.params
        trial_params["value"] = simulation(model.predict(dr),r)# Nouvelle valeur de learn_period
        trial.update_attrs(params=trial_params)

def actualisation_données(actu_étude=False,source="Binance",parent_folder = os.path.dirname(os.path.abspath(__file__))):
    for i in [f for f in os.listdir(os.path.join(parent_folder,source)) if os.path.isdir(os.path.join(parent_folder,source, f))]:
        for j in [f for f in os.listdir(os.path.join(parent_folder,source,i)) if os.path.isdir(os.path.join(parent_folder,source,i, f))]:
            actualiser_csv(i[:-5], j)
            if actu_étude:
                try:
                    actualisation_étude(i[:-5], j)
                except KeyError:
                    pass

def clear_étude(archive=True,source="Binance",parent_folder = os.path.dirname(os.path.abspath(__file__))):
    for i in [f for f in os.listdir(os.path.join(parent_folder,source)) if os.path.isdir(os.path.join(parent_folder,source, f))]:
        for j in [f for f in os.listdir(os.path.join(parent_folder,source,i)) if os.path.isdir(os.path.join(parent_folder,source,i, f))]:
            contenu=[f for f in os.listdir(os.path.join(parent_folder,source,i,j)) if os.path.isdir(os.path.join(parent_folder,source,i,j, f))]
            if archive:
                max=0
                for o in contenu:
                    if o[:len("archive")]=="archive":
                        if max<int(o.split(" ")[1]):
                            max=int(o.split(" ")[1])
                max+=1
                try:
                    os.mkdir(os.path.join(parent_folder,source,i,j,"archive "+str(max)))
                except FileExistsError:
                    pass
                for o in contenu:
                    try:
                        int(o)
                        os.rename(os.path.join(parent_folder,source,i,j,o),os.path.join(parent_folder,source,i,j,"archive "+str(max),o))
                    except ValueError:
                        pass
            contenu=[f for f in os.listdir(os.path.join(parent_folder,source,i,j)) if os.path.isdir(os.path.join(parent_folder,source,i,j, f))]
            rmtree(os.path.join(parent_folder,source,i,j,i+j+".db"))
            for o in contenu:
                try:
                    int(o)
                    rmtree(os.path.join(parent_folder,source,i,j, o))
                except ValueError:
                    pass

def lire(chemin):
    with open (chemin,"r") as fichier:
        return load(fichier)

def enregistrer (chemin,contenu,modeLecture="w"):
    with open(chemin,modeLecture) as fichier :
        d(contenu,fichier,indent=4, ensure_ascii=False, sort_keys=False )

def envoie_I_A_(source="Binance",parent_folder = os.path.dirname(os.path.abspath(__file__))):
    for i in [f for f in os.listdir(os.path.join(parent_folder,source)) if os.path.isdir(os.path.join(parent_folder,source, f))]:
        for j in [f for f in os.listdir(os.path.join(parent_folder,source,i)) if os.path.isdir(os.path.join(parent_folder,source,i,f))]:
            try:
                study = load_study(study_name=i+"-"+j, storage="sqlite:///"+os.path.join(parent_folder,source, i+"-USDT", j, i+"-"+j+".db"),sampler=TPESampler())
                df = study.best_trial.number
                learn_period = int(study.best_trial["learn_period"])
                while True:
                    choix=input(i,j,df)
                    if choix in ["y","yes","oui","o"]:
                        copytree(os.path.join(parent_folder,source,i,j,df),os.path.join(os.path.dirname(os.path.split(parent_folder)[0]),"Actif",source,i,j,df))
                        try:
                            dico=lire(os.path.join(os.path.dirname(os.path.split(parent_folder)[0]),"Actif",source,i,j,"paramètre.json"))
                            dico["learn_period"]=learn_period
                        except FileNotFoundError:
                            dico={"learn_period": learn_period}
                        enregistrer(os.path.join(os.path.dirname(os.path.split(parent_folder)[0]),"Actif",source,i,j,"paramètre.json"),dico)
                        print(os.path.join(os.path.dirname(os.path.split(parent_folder)[0]),"Actif",source,i,j,df))
                    elif choix in ["n","no","non","n"]:
                        pass
                    else:
                        print("Erreur de compréhension !")
                        continue
            except KeyError:
                pass



