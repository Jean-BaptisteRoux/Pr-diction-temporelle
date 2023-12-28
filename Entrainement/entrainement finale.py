import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import optuna,math,time
from sklearn.preprocessing import MinMaxScaler

# Chargement des données
df = pd.read_csv("/home/Astra_world/Data/Binance/1h/MATIC-USDT.csv")
df['date'] = pd.to_datetime(df['date'], unit='ms')
df=df.set_index('date')
dataset=df.filter(["close"]).values
training_data_len=math.ceil(len(dataset))

# Normalisation des données
scaler=MinMaxScaler(feature_range=(0,1))
scaled_data=scaler.fit_transform(dataset)

def données(scaler,dataset):
    scaled_data=scaler.transform(dataset)
    dataset=df[:"2022"].filter(["close"]).values
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
X, y = données(scaler,df[:"2022"].filter(["close"]).values)
dr,r=données(scaler,df["2023"].filter(["close"]).values[7:])

# Fonction d'objectif pour l'optimisation bayésienne
def objective(trial):
    # Définition des hyperparamètres à optimiser
    dropout = trial.suggest_uniform("dropout", 0, 0.5)
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-1)
    num_units = trial.suggest_int("num_units", 32, 512)
    num_layers = trial.suggest_int("num_layers", 1, 5)
    bias_initializer = trial.suggest_categorical("bias_initializer", ["zeros", "ones", "random_normal", "random_uniform"])
    kernel_initializer = trial.suggest_categorical("kernel_initializer", ["glorot_uniform", "glorot_normal", "he_normal", "he_uniform"])
    activation = trial.suggest_categorical("activation", ["relu", "sigmoid", "tanh"])
    units = trial.suggest_int("units", 32, 512)
    recurrent_dropout = trial.suggest_uniform("recurrent_dropout", 0, 0.5)
    epochs = trial.suggest_int("epochs", 1,75)
    batch_size = trial.suggest_int("batch_size", 16, 128)
    
    # Entrainement et prédiction
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Définition du modèle LSTM
    model = keras.Sequential()
    for i in range(num_layers):
        model.add(layers.LSTM(num_units, dropout=dropout, recurrent_dropout=recurrent_dropout, bias_initializer=bias_initializer, kernel_initializer=kernel_initializer, return_sequences=True, input_shape = (X_train.shape[1], 1)))
    model.add(layers.Dense(units, activation=activation))
    model.add(layers.Dropout(dropout))
    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation=activation))

    # Compilation du modèle
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    e=model.compile(loss="mse", optimizer=optimizer)



    with tf.device('/device:GPU:0'):

    # Entrainement du modèle
        history=model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val), verbose=1)
        print(history.history)    # perte sur l'ensemble de validation

    # Prédiction sur les données de validation
        y_pred = model.predict(X_val)
    predictions=y_pred
    predictions = np.reshape(predictions, (predictions.shape[0], predictions.shape[1]))
    predictions = scaler.inverse_transform(predictions)
    # Calcul de la métrique à optimiser (RMSE)
    mse = mean_squared_error(y_val, y_pred)
    print(mse)

    global dr,r
    predictions = model.predict(dr)
    reel=r
    dollard,btc=100,0
    for i in range(1,len(predictions)):
        if predictions[i]>reel[i-1] and dollard:
            btc=(dollard/reel[i-1])*0.999
            dollard=0
        elif predictions[i]>reel[i-1] and btc:
            dollard=(btc*reel[i-1])*0.999
            btc=0
    dollard+=(btc*reel[i])
    """y_pred_binary,c=0,0
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
        pass"""
    with open("/home/Astra_world/Astra_investissement/Entrainement/best dollard.txt","r") as best:
        b=best.readlines()
    if float(b[-1].split(",")[0])<dollard:
        model.save("/home/Astra_world/Astra_investissement/Entrainement/I.A dollard")
    with open("/home/Astra_world/Astra_investissement/Entrainement/best dollard.txt","a") as best:
        best.write(str(dollard)+"\n")


    return float(dollard)

#Lancement de l'optimisation bayésienne
study_name = 'MATIC-USDT1hdollard'
direction=["maximize","minimize"]

try:
    study = optuna.create_study(study_name=study_name, storage="sqlite:///"+study_name+".db",direction=direction[0])
except :
    study = optuna.load_study(study_name=study_name, storage="sqlite:///"+study_name+".db", sampler=optuna.samplers.TPESampler())
study.optimize(objective, n_trials=100)
#Récupération des meilleurs hyperparamètres

best_params = study.best_params
#study.trials_dataframe().to_csv('/home/Astra_world/Astra_investissement/Entrainement/Résultat '+study_name+' '+time.ctime()+'.csv', index=False)
print(study.trials_dataframe())
print(f"Meilleurs hyperparamètres : {best_params}")
#Entrainement du modèle avec les meilleurs hyperparamètres
"""x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
X, y = x_train, y_train"""
model = keras.Sequential()
for i in range(best_params["num_layers"]):
    model.add(layers.LSTM(
        best_params["num_units"],
        dropout=best_params["dropout"],
        return_sequences=True,
        bias_initializer=best_params['bias_initializer'],
        kernel_initializer=best_params['kernel_initializer'],
        recurrent_dropout=best_params['recurrent_dropout'],
        activation=best_params['activation']
    ))
model.add(layers.Dense(best_params["units"], activation=best_params["activation"]))
model.add(layers.Dropout(best_params["dropout"]))
model.add(layers.Flatten())
model.add(layers.Dense(1))

optimizer = keras.optimizers.Adam(learning_rate=best_params["learning_rate"])
model.compile(loss="mse", optimizer=optimizer)

model.fit(X, y, epochs=best_params["epochs"], batch_size=best_params["batch_size"], verbose=1)
model.save("/home/Astra_world/Astra_investissement/Entrainement/I.A")
print("Modèle entrainé et sauvegardé avec succès")
dataset=df["2022"].filter(["close"]).values
training_data_len=math.ceil(len(dataset))

# Normalisation des données
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
X, y = x_train, y_train
y_pred = model.predict(X)
predictions=y_pred
predictions = np.reshape(predictions, (predictions.shape[0], predictions.shape[1]))
predictions = scaler.inverse_transform(predictions)
# Calcul de la métrique à optimiser (RMSE)
print(np.sqrt(mean_squared_error(y, y_pred)))

reel=df["2022"].filter(["close"]).values[7:]
dollard,btc=100,0
for i in range(1,len(predictions)):
    if predictions[i]>reel[i] and dollard:
        btc=(dollard/reel[i])*0.999
        dollard=0
    elif predictions[i]>reel[i] and btc:
        dollard=(btc*reel[i])*0.999
        btc=0
dollard+=(btc*reel[i])
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
print(np.sqrt(mean_squared_error(y, y_pred)),dollard,y_pred_binary)
