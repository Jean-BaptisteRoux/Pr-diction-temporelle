from fonction import nouvelle_étude_binance

# Chemin relatif du dossier parent (où les dossiers seront créés)

# Liste des noms de dossiers à créer
crypto = ["BTC", "ETH", "XRP","AAVE","ADA","AVAX","BCH","BNB","DOGE","EGLD","FTM","LINK","LTC","LUNA","MATIC","NEAR","SOL","UNI","XTZ"]
interval_list = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M"]

# Création des dossiers
for folder in crypto:
    for folder2 in interval_list:
        nouvelle_étude_binance(folder,folder2)
        
            

