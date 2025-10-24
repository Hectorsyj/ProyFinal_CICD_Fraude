import pandas as pd
import numpy as np
import gdown
import pickle
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

file_id = '1lxQ4x04vIXwhGERd3YBh55bEixlnsimQ'
url = f'https://drive.google.com/uc?id={file_id}'
gdown.download(url, 'creditcard.csv', quiet=False)
data = pd.read_csv('creditcard.csv')

# Datos fuera del estado normal
colNum = data.select_dtypes(include=['float']).columns
colOth = data.columns.difference(colNum)
print(colOth)
dfEl = data[colNum].copy()
Q1 = dfEl.quantile(0.25)
Q3 = dfEl.quantile(0.75)
IQR = Q3 - Q1

dfEl = dfEl[(dfEl >= Q1 - 1.5*IQR) & (dfEl <= Q3 + 1.5*IQR)]
dfEl[colOth] = data[colOth]
dfEl = dfEl.dropna()

imputer = KNNImputer(n_neighbors=5)
df_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

print(f'datos antes de elimiinar {len(data)} y despues de eso {len(dfEl)} eliminando un {1- len(dfEl)/len(data):.4}')

#Analisis de balance de clases
bal = data['Class'].value_counts()
print(f"{bal} \n")

for i in bal:
  print(f'{round((i/bal.sum())*100,2)}%')

#Normalizacion de datos
x = data.drop(['Class','Time'], axis=1)
y = data['Class']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(x)

with open('data/X_scaled.pkl', 'wb') as f:
    pickle.dump(X_scaled, f)

with open('data/y.pkl', 'wb') as f:
    pickle.dump(y, f)
    
print("preprocesamiento terminado")