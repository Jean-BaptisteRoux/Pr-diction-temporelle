# %%
import asyncio,time,json,requests,csv

from decimal import *
from binance.client import AsyncClient


getcontext().prec = 3


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
paramètres=lire("/home/backtest_tools_Entrainement_I_A_/paramètre.json")
with open(paramètres["chemin_pair"],"r") as p:
    pair=p.read()

async def main(pair):
    api_key=paramètres["api_key"]
    api_secret=paramètres["api_secret"]
    logs("Connection avec Binance en cours")
    client =  await AsyncClient.create(api_key,api_secret)
    while True:
        matic_price= requests.get("https://api.binance.com/api/v3/ticker/price?symbol="+str(pair[0])+str(pair[1])).json()['price']
        logs("Connection avec Binance effectué",1)
        logs("Soldes en cours de téléchargement")
        solde_pair_base = await client.get_asset_balance(pair[1])
        solde_pair_base=Decimal(solde_pair_base["free"])
        solde_pair= await client.get_asset_balance(pair[0])
        solde_pair=Decimal(solde_pair["free"])
        logs("Soldes récupérés",1)
        logs(str(pair[0])+" : "+str(solde_pair),False)
        logs(str(pair[1])+" : "+str(solde_pair_base),False)
        fichier=[str(time.perf_counter_ns()),str(solde_pair)*matic_price,str(solde_pair_base)/matic_price]
        with open(paramètres["compte"],"a") as c:
            truc=csv.writer(c)
            truc.writerow(fichier)
        print(fichier)
        await asyncio.sleep(1)

    
    logs("Fermeture de la session en cours")
    await client.close_connection()
    logs("Session fermée")

if __name__ == "__main__":

    loop = asyncio.get_event_loop()
    loop.run_until_complete(main(pair.split("/")))
