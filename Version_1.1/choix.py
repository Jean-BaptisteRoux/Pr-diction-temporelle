# %%
import asyncio,time,json,requests

from binance.client import AsyncClient
from os import getcwd


def lire(chemin):
    with open (chemin,"r") as fichier:
        return json.load(fichier)

def logs(message,niveau=0,affi=True):
    for i in range (niveau+1):
        with open(paramètres["chemin_logs"]+"logs"+str(i)+".txt","a") as l :
            l.write(time.ctime()+" : "+message+"\n")
    if affi:
        print(str(time.gmtime()[3])+":"+str(time.gmtime()[4])+":"+str(time.gmtime()[5])+" "+str(message))

def enregistrer (chemin,contenu,modeLecture="w"):
    with open(chemin,modeLecture) as fichier :
        json.dump(contenu,fichier,indent=4, ensure_ascii=False, sort_keys=False )
# %%
paramètres=lire("/home/Astra_world/Astra_investissement/paramètre.json")
clés=lire("/home/Astra_world/Astra_investissement/keys.json")
predi,reel=[],[]
with open(paramètres["chemin_predi"],"r") as p:
    predi=p.read()
with open(paramètres["chemin_reel"],"r") as p:
    reel=p.read()
with open(paramètres["chemin_pair"],"r") as p:
    pair=p.read()

async def main(pair):
    api_key=clés["api_key"]
    api_secret=clés["api_secret"]
    logs("Connection avec Binance en cours")
    client =  await AsyncClient.create(api_key,api_secret)
    matic_price= float(requests.get("https://api.binance.com/api/v3/ticker/price?symbol=MATICUSDT").json()['price'])
    logs("Connection avec Binance effectué",1)
    logs("Soldes en cours de téléchargement")
    solde_pair_base = await client.get_asset_balance(pair[1])
    solde_pair_base=round(float(solde_pair_base["free"]),5)
    solde_pair= await client.get_asset_balance(pair[0])
    solde_pair=round(float(solde_pair["free"]),5)
    logs("Soldes récupérés",1)
    logs(str(pair[0])+" : "+str(solde_pair),False)
    logs(str(pair[1])+" : "+str(solde_pair_base),False)
    achat=(solde_pair_base/matic_price)//1
    pair=pair[0]+pair[1]
    if predi>reel:
        logs("Je prédis "+str(predi)+" qui est plus grand que "+str(reel)+" qui est le prix actuel.",2)
        if solde_pair_base>10.3 :
            logs("J'investi "+str(achat)+" !",3)
            #order=await client.create_order(symbol=str(pair),side="BUY",quantity=achat,type="MARKET")
            if order:
                dorder=lire(paramètres["chemin_order"])
                dorder[time.asctime(time.gmtime())]=order
                print(dorder[time.asctime(time.gmtime())])
                enregistrer(paramètres["chemin_order"],dorder)
        else:
            logs("Malheureusement je ne peux pas faute de moyen")
    elif predi<reel:
        logs("Je prédis "+str(predi)+" qui est plus petit que "+str(reel)+" qui est le prix actuel.",2)
        if solde_pair>10.3:
            logs(str(solde_pair)+" "+str(round(matic_price,5)),1)
            logs("J'enlève "+str((solde_pair)//1)+" !",3)
            #order=await client.create_order(symbol=str(pair),side="SELL",quantity=solde_pair//1,type="MARKET")
            if order:
                dorder=lire(paramètres["chemin_order"])
                dorder[time.asctime(time.gmtime())]=order
                print(dorder[time.asctime(time.gmtime())])
                enregistrer(paramètres["chemin_order"],dorder)
        else:
            logs("Malheureusement je peux pas faute de moyen")
        
    
    logs("Fermeture de la session en cours")
    await client.close_connection()
    logs("Session fermée")

if __name__ == "__main__":

    loop = asyncio.get_event_loop()
    loop.run_until_complete(main(pair.split("/")))
