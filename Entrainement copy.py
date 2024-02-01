from os import environ
environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from optuna.logging import set_verbosity,ERROR 
set_verbosity(ERROR)
#from fonction import lancement_des_études,actualisation_données
from fonction_copy import *
import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText
import threading
import time
from tqdm import tqdm
from Evenement import Modif_fichier
from watchdog.observers import Observer
from tkinter import scrolledtext, Tk, END

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        
        
        # Initialisation de la fenêtre
        self.title("Interface graphique")
        self.geometry("600x400")

        # Création des barres de progression
        self.progress_bar1 = ttk.Progressbar(self, orient="horizontal", length=300, mode="determinate")
        self.progress_bar2 = ttk.Progressbar(self, orient="horizontal", length=300, mode="determinate")
        self.progress_bar3 = ttk.Progressbar(self, orient="horizontal", length=300, mode="determinate")
        self.progress_bar1.pack(pady=10)
        self.progress_bar2.pack(pady=10)
        self.progress_bar3.pack(pady=10)
        self.pb1, self.pb2, self.pb3 = [300, 300, 300]


        # Création de la zone de texte
        self.text_area = ScrolledText(self, height=10, width=50)
        self.text_area.pack(pady=10)
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"Binance.txt"), 'r') as f:
            content = f.read()
        self.text_area.insert(END, content)

        # Création des boutons
        self.start_button = tk.Button(self, text="Démarrer", command=self.start_thread)
        self.enregistre_button = tk.Button(self, text="Enregistre", command=self.enregistre)
        self.stop_button = tk.Button(self, text="Arrêter", command=self.stop_thread, state="disabled")
        self.pause_button = tk.Button(self, text="Mettre en pause", command=self.pause_thread, state="disabled")
        self.start_button.pack(side="left", padx=10)
        self.stop_button.pack(side="left", padx=10)
        self.pause_button.pack(side="left", padx=10)
        self.enregistre_button.pack(side="left", padx=10)

        # Initialisation des variables
        self.thread = None
        self.running = False
        self.paused = False

        # Initialisation de l'évenement
        event_handler = Modif_fichier(os.path.join(os.path.dirname(os.path.abspath(__file__)),"Binance.txt"), self.text_area)
        self.oeil = Observer()
        self.oeil.schedule(event_handler, os.path.dirname(os.path.join(os.path.dirname(os.path.abspath(__file__)),"Binance.txt")), recursive=False)
        self.oeil.start()

    def enregistre(self):
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"Binance.txt"),"w") as b:
            b.write(self.text_area.get("1.0", "end")[:-1])
            print(self.text_area.get("1.0", "end")[:-1])

    def start_thread(self):
        self.running = True
        self.thread = threading.Thread(target=self.my_function)
        self.thread.start()
        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.pause_button.config(state="normal", text="Mettre en pause")

    def stop_thread(self):
        self.running = False
        self.progress_bar1["value"] = 0
        self.progress_bar2["value"] = 0
        self.progress_bar3["value"] = 0
        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.pause_button.config(state="disabled", text="Mettre en pause")
    

    def pause_thread(self):
        self.paused = not self.paused
        if self.paused:
            self.pause_button.config(text="Reprendre")
        else:
            self.pause_button.config(text="Mettre en pause")
    
    def my_function(self):
        # Exemple de fonction
        while self.running:
            lancement_des_études(self)


"""# créer et démarrer le thread
        thread = threading.Thread(target=self.my_function)"""
app = App()
try:
    app.mainloop()
finally:
    app.oeil.stop()
    app.oeil.join()
