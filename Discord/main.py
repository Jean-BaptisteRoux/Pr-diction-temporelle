import time
from os import system
retour=1
while retour!=0:
    retour=system("python3 vmain.py")
    if retour!=0:
        time.sleep(60)