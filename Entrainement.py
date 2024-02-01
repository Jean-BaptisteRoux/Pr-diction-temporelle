from os import environ
environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from optuna.logging import set_verbosity,ERROR 
set_verbosity(ERROR)
#from fonction import lancement_des_études,actualisation_données
from fonction import *

lancement_des_études()