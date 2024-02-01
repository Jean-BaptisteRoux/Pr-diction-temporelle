from watchdog.events import FileSystemEventHandler
from tkinter import  END

class Modif_fichier(FileSystemEventHandler):
    """Classe pour gérer les événements de modification de fichier"""

    def __init__(self, filename, text_widget):
        super().__init__()
        self.filename = filename
        self.text_widget = text_widget

    def on_modified(self, event):
        if event.src_path == self.filename:
            self.update_text_widget()

    def update_text_widget(self):
        """Fonction pour mettre à jour le contenu de la ScrolledText"""
        with open(self.filename, 'r') as f:
            content = f.read()
        self.text_widget.delete('1.0', END)
        self.text_widget.insert(END, content)