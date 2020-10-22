# ipln2020

Se utilizó Support Vector Machine (de scikit learn) en python 3.7.3 para predecir las clasificaciones de los tweets.

Se leen los tweets provistos, al igual que los embeddings, utilizando pandas. Cada tweet es representado por el promedio de los embeddings de las palabras que lo componen. Dichos promedios son la entrada de la SVM. En el caso en que se encuentra una palabra desconocida que no tiene un embedding correspondiente entonces se sortea un vector de 300 números aleatorios entre -1 y 1.

Tuneamos hiperparámetros utilizando Grid Search y el conjunto de validación y nos quedamos con los valores devueltos.

Se imprimen las siguientes métricas: accuracy, precision, recall y f1.

Llamada al programa:
`python es_humor.py <PATH_TO_CORPUS> <test1.csv> <test2.csv> ... <testN.csv>`
