import json

def lire(chemin):
    with open (chemin,"r") as fichier:
        return json.load(fichier)

def enregistrer (chemin,contenu,modeLecture="w"):
    with open(chemin,modeLecture) as fichier :
        json.dump(contenu,fichier,indent=4, ensure_ascii=False, sort_keys=False )
    
while True:
    dico=lire("/home/backtest_tools_Entrainement_I_A_/paramètre.json")
    clé=input("Clé : ")
    porte=input("Contenu : ")
    dico[clé]=porte
    enregistrer("/home/backtest_tools_Entrainement_I_A_/paramètre.json",dico)