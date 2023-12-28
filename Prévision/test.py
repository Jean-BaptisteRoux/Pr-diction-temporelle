from binance.client import Client
import json

def lire(chemin):
    with open (chemin,"r") as fichier:
        return json.load(fichier)

# Créez une instance du client en utilisant votre clé d'API Binance
clés=lire("/home/Astra_world/Astra_investissement/keys.json")
api_key=clés["api_key"]
api_secret=clés["api_secret"]

# Connexion à Binance avec votre clé API
client = Client(api_key, api_secret)

deposit_history = client.get_deposit_history()

print(deposit_history)
