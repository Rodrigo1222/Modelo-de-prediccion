import pandas as p
from sklearn.tree import DecisionTreeClassifier
import joblib #persistir modelo o leerlos despues 

#preparar la Data
data_juegos = p.read_csv("C:/Users/paz_i/workspace/Machine Learning/juegos-ml.csv")
X = data_juegos.drop(columns=["juegos"])
y = data_juegos["juegos"]

#Cargar la data
modelo = DecisionTreeClassifier()
modelo.fit(X.values, y)

joblib.dump(modelo, "recomendador_juegos.joblib")
