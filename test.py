from fonction import *
import optuna.visualization as vis
from optuna import load_study
from optuna.samplers import TPESampler

study = load_study(study_name="BCH-USDT1h", storage="sqlite:////home/Astra_world/Astra_investissement23/Ferme/Binance/BCH-USDT/1h/BCH-USDT1h.db",sampler=TPESampler())

def simulation(predictions,reel,taux=1-0.999):
    """Cette fonction simule l'achat et le vente d'un actif selon la différence entre predictions[i] et reel[i]"""
    dollard=[100]
    for i in range(1,len(predictions)): #On part de 1 pour éviter une erreur à i=0
        #print(predictions[i],reel[i])
        if predictions[i]>reel[i] :#On achète
            dollard.append((dollard[-1]/reel[i])*(1-taux))
        elif predictions[i]<reel[i] :#On vend
            dollard.append((dollard[-1]/reel[i])*(1-taux))
    print(dollard,(100*reel[-1]/reel[0]),reel[-1],reel[0])
    """dollard-=(100*reel[-1]/reel[0])"""
    print("Résultat brut : "+str(dollard))
    return dollard
from optuna.visualization import plot_parallel_coordinate

import matplotlib.pyplot as plt
import numpy as np

def plot_multiple_curves(data):
    # Générer les couleurs pour chaque courbe
    num_curves = len(data)
    colors = plt.cm.rainbow(np.linspace(0, 1, num_curves))

    # Créer la figure et les axes
    fig, ax = plt.subplots()

    # Parcourir les données et tracer les courbes avec les couleurs correspondantes
    for i, curve_data in enumerate(data):
        ax.plot(curve_data, color=colors[i])

    # Afficher la légende avec les couleurs correspondantes
    legend_labels = [f"Courbe {i+1}" for i in range(num_curves)]
    ax.legend(legend_labels)

    # Afficher le graphique
    plt.show()


#plot_parallel_coordinate(study).show()
liste=[]
for j in range(15):
    try:
        model=load_model("/home/Astra_world/Astra_investissement23/Ferme/Binance/AAVE-USDT/1h/"+str(j))
        for i in range(100):
            try:
                df=data("AAVE-USDT","1h")
                X_train,valid,dr,r,scaler=données_séparées(df,i,2)
                reel=df.loc["2023":].filter(["close"])
                #print(reel.iloc[-1].values , reel.iloc[0].values,reel.iloc[-1].values / reel.iloc[0].values)
                courbe=100*(reel.iloc[-1].values / reel.iloc[0].values)
                x,y,dr,r,scaler=données_séparées(df,i,2)
                liste.append(simulation(scaler.inverse_transform(model.predict(dr).reshape(-1, 1)).reshape(-1),scaler.inverse_transform(r.reshape(-1, 1)).reshape(-1)))

                break
            except ValueError:
                pass
    except OSError:
        pass
plot_multiple_curves(liste)