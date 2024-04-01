
#consulta con un un modelo ya establecido

import pandas as p
from sklearn.tree import DecisionTreeClassifier
import joblib

modelo = joblib.load("recomendador_juegos.joblib")
predicciones = modelo.predict([[14,0]])
predicciones

