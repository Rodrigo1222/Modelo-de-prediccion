import pandas as p
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#preparar la Data
data_juegos = p.read_csv("C:/Users/paz_i/workspace/Machine Learning/juegos-ml.csv")
X = data_juegos.drop(columns=["juegos"])
y = data_juegos["juegos"]

#Datos de prueba y de entrenamiento
X_entrenar, X_prueba, y_entrenar, y_prueba = train_test_split(X.values, y, test_size=0.2)
#test_size es la division de los datos se recomienda 80% datos para entrenarlo y 20% para probar los datos  

#Cargar la data
modelo = DecisionTreeClassifier()
modelo.fit(X_entrenar, y_entrenar)
predicciones = modelo.predict(X_prueba)

#comparaccion de datos de prueba con las predicciones
puntaje = accuracy_score(y_prueba, predicciones)
 # un valor entre 0 a 1 donde cero es completamente erronea y 1 es completamente certero
puntaje
