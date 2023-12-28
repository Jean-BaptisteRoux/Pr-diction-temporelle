import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.constraints import MaxNorm
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import optuna, math, time,os
from sklearn.preprocessing import MinMaxScaler
from plyer import notification


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




# Fonction d'objectif pour l'optimisation bayésienne
def objective(trial):
    # Définition des hyperparamètres à optimiser
    clipvalue = trial.suggest_uniform('clipvalue', 0.01, 1.0)
    dropout = trial.suggest_uniform("dropout", 0, 0.5)
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-1)
    num_units = trial.suggest_int("num_units", 32, 512)
    num_layers = trial.suggest_int("num_layers", 1, 5)
    bias_initializer = trial.suggest_categorical("bias_initializer", ["zeros", "ones", "random_normal", "random_uniform"])
    kernel_initializer = trial.suggest_categorical("kernel_initializer", ["glorot_uniform", "glorot_normal", "he_normal", "he_uniform"])
    activation = trial.suggest_categorical("activation", ["relu", "sigmoid", "tanh"])
    units = trial.suggest_int("units", 32, 512)
    recurrent_dropout = trial.suggest_uniform("recurrent_dropout", 0, 0.5)
    epochs = trial.suggest_int("epochs", 1, 150)
    batch_size = trial.suggest_int("batch_size", 16, 128)
    l1_lambda = trial.suggest_loguniform("l1_lambda", 1e-5, 1e-1)
    l2_lambda = trial.suggest_loguniform("l2_lambda", 1e-5, 1e-1)

    # Entrainement et prédiction
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

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
                            input_shape=(X_train.shape[1], 1)))
    model.add(layers.Dense(units,activation=activation, kernel_constraint=MaxNorm(clipvalue)))
    model.add(layers.Flatten())
    model.add(layers.Dense(1,activation=activation, kernel_constraint=MaxNorm(clipvalue)))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                loss=keras.losses.MeanSquaredError(),
                metrics=[keras.metrics.RootMeanSquaredError()])
    print("Début de l'entrainement")
    start_time = time.time()
    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val), verbose=1)
    end_time = time.time()
    print("Fin de l'entrainement")

    # Prédiction sur les données de validation
    y_pred = model.predict(X_val)
    # Inversion de la normalisation des données
    y_pred = scaler.inverse_transform(y_pred)
    y_val = scaler.inverse_transform(y_val.reshape(-1, 1))
    # Calcul des métriques d'évaluation
    try:
        rmse = mean_squared_error(y_val, y_pred, squared=False)
        r2 = r2_score(y_val, y_pred)
    except: 
        pass
    # Affichage des résultats
    print(f"time={end_time-start_time:.2f}s, "
        f"dropout={dropout:.2f}, learning_rate={learning_rate:.5f}, "
        f"num_units={num_units}, num_layers={num_layers}, bias_initializer={bias_initializer}, "
        f"kernel_initializer={kernel_initializer}, activation={activation}, units={units}, "
        f"recurrent_dropout={recurrent_dropout:.2f}, epochs={epochs}, batch_size={batch_size}, "
        f"l1_lambda={l1_lambda:.5f}, l2_lambda={l2_lambda:.5f}")
    global dr,r
    predictions = model.predict(dr)
    reel=r
    dollard,btc=100,0
    for i in range(1,len(predictions)):
        if predictions[i]>reel[i-1] and dollard:
            btc=(dollard/reel[i-1])*0.999
            dollard=0
        elif predictions[i]<reel[i-1] and btc:
            dollard=(btc*reel[i-1])*0.999
            btc=0
    dollard+=(btc*reel[i])
    # Retour de la valeur à optimiser (ici, le dollard)
    """with open("/home/Astra_world/Astra_investissement/Entrainement/best dollard gru.txt", "r") as f:
        values = [float(line.strip()) for line in f]"""
    if dollard!=100:
        if not os.path.exists("/home/Astra_world/Astra_investissement/Entrainement/Study/"+str(study.study_name)):
            os.makedirs("/home/Astra_world/Astra_investissement/Entrainement/Study/"+str(study.study_name))
        if not os.path.exists("/home/Astra_world/Astra_investissement/Entrainement/Study/"+study.study_name+"/"+str(trial.number)):
            os.makedirs("/home/Astra_world/Astra_investissement/Entrainement/Study/"+study.study_name+"/"+str(trial.number))
        model.save("/home/Astra_world/Astra_investissement/Entrainement/Study/"+study.study_name+"/"+str(trial.number))
    with open("/home/Astra_world/Astra_investissement/Entrainement/"+study.study_name+".txt","a") as best:
        best.write(str(dollard)+"\n")
    return dollard

#Optimisation bayésienne
#Lancement de l'optimisation bayésienne
path="/home/Astra_world/Data/Binance/1h"
for i in [os.path.join(path, file) for file in os.listdir(path) if file.endswith(".csv")]:
    # Chargement des données
    df = pd.read_csv(i)
    df['date'] = pd.to_datetime(df['date'], unit='ms')
    df=df.set_index('date')
    dataset=df.filter(["close"]).values
    training_data_len=math.ceil(len(dataset))
    # Normalisation des données
    scaler=MinMaxScaler(feature_range=(0,1))
    scaler.fit_transform(dataset)



    X, y = données(scaler,df[:"2022"].filter(["close"]).values)
    dr,r = données(scaler,df["2023"].filter(["close"]).values[7:])
    print(dr[-1],r[-1])
    study_name = i[len(path)+1:-4]
    direction=["maximize","minimize"]

    try:
        study = optuna.create_study(study_name=study_name, storage="sqlite:///"+study_name+".db",direction=direction[0])
    except :
        study = optuna.load_study(study_name=study_name, storage="sqlite:///"+study_name+".db", sampler=optuna.samplers.TPESampler())
    c=0
    while c<5:
        try:
            # Votre code ici
            print(df)
            print(study.trials_dataframe())
            """#study.trials_dataframe().to_csv('/home/Astra_world/Astra_investissement/Entrainement/Résultat '+study_name+' '+time.ctime()+'.csv', index=False)
            """# Récupérer la liste des essais échoués
            failed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]

            # Réenfiler les essais échoués
            for trial in failed_trials:
                study.enqueue_trial(trial.params)
                
            # Continuer l'optimisation
            study.optimize(objective, n_trials=len(failed_trials))
            if 100-len(study.trials)>0:
                study.optimize(objective, n_trials=100-len(study.trials))
            print("Le code s'est exécuté avec succès !")
            notification.notify(
                title="Succès !",
                message="Le code s'est exécuté avec succès !",
                app_name="Mon script",
            )
            break
        except Exception as e:
            # Gestion des erreurs
            print("Une erreur est survenue :", str(e))
            notification.notify(
                title="Erreur !",
                message=f"Une erreur est survenue :\n{str(e)}",
                app_name="Mon script",
            )
            time.sleep(10)
            c+=1

#Récupération des meilleurs hyperparamètres

"""best_params = study.best_params
study.trials_dataframe().to_csv('/home/Astra_world/Astra_investissement/Entrainement/Résultat '+study_name+' '+time.ctime()+'.csv', index=False)
print(study.trials_dataframe())
print(f"Meilleurs hyperparamètres : {best_params}")
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
model = keras.Sequential()
for i in range(best_params['num_layers']):
    model.add(layers.GRU(best_params['num_units'],
                        dropout=best_params['dropout'],
                        recurrent_dropout=best_params['recurrent_dropout'],
                        bias_initializer=best_params['bias_initializer'],
                        kernel_initializer=best_params['kernel_initializer'],
                        activation=best_params['activation'],
                        kernel_regularizer=keras.regularizers.L1L2(l1=best_params['l1_lambda'], l2=best_params['l2_lambda']),
                        return_sequences=True if i < best_params['num_layers'] - 1 else False,
                        input_shape=(X_train.shape[1], 1)))
model.add(layers.Dense(best_params['units'], activation=best_params['activation'], kernel_constraint=MaxNorm(best_params['clipvalue'])))
model.add(layers.Flatten())
model.add(layers.Dense(1, activation=best_params['activation'], kernel_constraint=MaxNorm(best_params['clipvalue'])))
model.compile(optimizer=keras.optimizers.Adam(learning_rate=best_params['learning_rate']),
            loss=keras.losses.MeanSquaredError(),
            metrics=[keras.metrics.RootMeanSquaredError()])
#history = model.fit(X, y, batch_size=best_params['batch_size'], epochs=best_params['epochs'], verbose=1)
regressor =load_model("/home/Astra_world/Astra_investissement/Entrainement/I.A dollard gru")
predictions = scaler.inverse_transform(model.predict(dr))
reel=scaler.inverse_transform(r.reshape(-1, 1))
dollard,btc=100,0
for i in range(0,len(predictions)):
    print(predictions[i],reel[i-1],dollard,btc)
    if predictions[i]>reel[i-1] and dollard:
        btc=(dollard/reel[i-1])*0.999
        dollard=0
    elif predictions[i]<reel[i-1] and btc:
        dollard=(btc*reel[i-1])*0.999
        btc=0
dollard+=(btc*reel[i])
with open("/home/Astra_world/Astra_investissement/Entrainement/best dollard gru.txt", "r") as f:
    values = [float(line.strip()) for line in f]
if int(max(values)) <dollard:
    model.save("/home/Astra_world/Astra_investissement/Entrainement/I.A dollard gru")
with open("/home/Astra_world/Astra_investissement/Entrainement/best dollard gru.txt","a") as best:
    best.write(str(dollard)+"\n")"""