import asyncio,csv
from binance.client import AsyncClient,Client
import time,json


def lire(chemin):
    with open (chemin,"r") as fichier:
        return json.load(fichier)


def logs(message,niveau=0,affi=True):
    for i in range (niveau+1):
        with open(paramètres["chemin_logs"]+"logs"+str(i)+".txt","a") as l :
            l.write(time.ctime()+" : "+message+"\n")
    if affi:
        print(str(time.gmtime()[3])+":"+str(time.gmtime()[4])+":"+str(time.gmtime()[5])+" "+str(message))
# %%
paramètres=lire("/home/Astra_world/Astra_investissement/paramètre.json")
clés=lire("/home/Astra_world/Astra_investissement/keys.json")
with open(paramètres["chemin_pair"],"r") as p:
    pair=p.read()

async def main(pair):
    api_key=clés["api_key"]
    api_secret=clés["api_secret"]
    logs("Connection avec Binance en cours")
    client = await AsyncClient.create(api_key,api_secret)
    logs("Connection avec Binance effectué")
    with open (paramètres["chemin_csv_maticusdt"],"r") as f:
        csv2=csv.reader(f, delimiter=',', quotechar='|')
        fichier=[]
        for i in csv2:
            fichier.append(i)
    del fichier[-1]
    del fichier[-1]
    # récupérer les données de prix pour le symbole MATICUSDT
    limite=((round(time.time_ns()/1000000)-int(fichier[-1][0]))//3600000)
    logs(str(limite)+" lignes à télécharger.")
    if limite:
        logs("Données en cours de téléchargement")
        klines = await client.get_klines(symbol=str(pair[0])+str(pair[1]), interval=Client.KLINE_INTERVAL_1HOUR, limit=limite)
        logs("Données téléchargées")
        # traiter les données klines
        logs("Écriture des données")
        with open (paramètres["chemin_csv_maticusdt"],"w") as f:
            truc=csv.writer(f)
            truc.writerows(fichier)
            for kline in klines:
                truc.writerow([kline[0],kline[1],kline[2],kline[3],kline[4],kline[5]])
        logs("Données écrites")
    logs("Fermeture de la session en cours")
    await client.close_connection()
    logs("Session fermée")

if __name__ == "__main__":

    loop = asyncio.get_event_loop()
    loop.run_until_complete(main(pair.split("/")))
