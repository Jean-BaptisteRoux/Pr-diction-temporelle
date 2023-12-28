import optuna.storages
import pandas as pd
from optuna.trial import TrialState
from optuna.distributions import *


df=pd.read_csv("/home/Astra_world/Astra_investissement/Entrainement/Résultat Sat Mar 18 01:36:58 2023.csv")

# Créer un objet Storage pour la base de données SQLite.
storage = optuna.storages.RDBStorage(url='sqlite:///MATIC-USDT1hdollard.db')

# Récupérer l'objet study de la base de données
study = optuna.load_study(storage=storage, study_name='MATIC-USDT1hdollard')

# Ajouter les résultats de la study
for i, row in df.iterrows():
    p = {
        'activation': row['params_activation'],
        'batch_size': row['params_batch_size'],
        'bias_initializer': row['params_bias_initializer'],
        'dropout': row['params_dropout'],
        'epochs': row['params_epochs'],
        'kernel_initializer': row['params_kernel_initializer'],
        'learning_rate': row['params_learning_rate'],
        'num_layers': row['params_num_layers'],
        'num_units': row['params_num_units'],
        'recurrent_dropout': row['params_recurrent_dropout'],
        'units': row['params_units']
    }
    trial = optuna.create_trial(
        value=row['value'],
        state=getattr(TrialState, row['state'].upper()),
        params=p,
        distributions={
            'activation': CategoricalDistribution([row['params_activation']]),
            'batch_size': IntUniformDistribution(row['params_batch_size'], row['params_batch_size']),
            'bias_initializer': CategoricalDistribution([row['params_bias_initializer']]),
            'dropout': UniformDistribution(row['params_dropout'], row['params_dropout']),
            'epochs': IntUniformDistribution(row['params_epochs'], row['params_epochs']),
            'kernel_initializer': CategoricalDistribution([row['params_kernel_initializer']]),
            'learning_rate': LogUniformDistribution(row['params_learning_rate'], row['params_learning_rate']),
            'num_layers': IntUniformDistribution(row['params_num_layers'], row['params_num_layers']),
            'num_units': IntUniformDistribution(row['params_num_units'], row['params_num_units']),
            'recurrent_dropout': UniformDistribution(row['params_recurrent_dropout'], row['params_recurrent_dropout']),
            'units': IntUniformDistribution(row['params_units'], row['params_units'])
        },
        user_attrs={
            'duration': row['duration'],
            'datetime_start': row['datetime_start'],
            'datetime_complete': row['datetime_complete']
        }
    )

    trial.distributions.update()
    
    study.add_trial(trial)



print(study.trials_dataframe())