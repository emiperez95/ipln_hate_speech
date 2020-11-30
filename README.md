# Es Odio

## Instalación

Usar el archivo de requerimientos provisto para instalar los modulos.
```bash
pip install requirements.txt
```
Si llegase a fallar la instalacion, se puede utilizar el archivo requirements_wide ,que contiene los modulos sin indicar la version requerida.
```bash
pip install requirements_wide.txt
```

## Uso

La ejecucion del programa se puede realizar mediante python
```bash
python es_odio.py
```
y para realizar la predicción de mas tweets, se pueden incluir archivos como argumentos al programa.
Si se deseara predecir los textos en los archivos ~/textos1.csv y ../../textos2.csv, 
```bash
python es_odio.py ~/textos1.csv ../../textos2.csv
```
Los resultados se escriben en un archivo con el mismo nombre que las entradas y se ubicaran en el mismo directorio.
En este caso se generarian los archivos ~/textos1.out y ../../textos2.out

## Trabajo realizado

### Análisis estadístico
### Preprocesamiento
Para modularizar el preprocesamiento, se decidio definir un conjunto de funciones que se aplican en un pipe de procesamiento. De esta forma es facil de visualizar cuales se aplican y de agregar o sacar en caso de ser necesario.

Para el parsing de texto, se crearon funciones que remueven urls, convierten saltos de lineas a espacios (tratamos cada tweet como oracion), se remueven caracteres especiales (de puntuacion y similares), se remueven caracteres ascii y se hace un strip de espacios redundantes.
Luego se realiza la tokenizacion de la "oracion" resultante. Dado que se quitaron todos los caracteres que podrian haber molestado, se decidio utilizar una tokenizacion simple separando segun espacios. A pesar de esto, se intento usar el tokenizador de ntlk, pero los resultados no fueron tan buenos.
Una vez tokenizadas las oraciones, se opto por eliminar las palabras de dos o menos letras y las stopwords del lenguaje. Estas dos operaciones se realizan para eliminar las palabras que aparecen mucho en las oraciones y que no aportan a la decision de discurso de odio.
Finalmente se aplico stemming y lemmatization. En el caso de stemming se encontraron dos librerias diferentes, Porter stemming que se basa en un corpus en ingles y Snowball stemming que es especifica para el español. Sobre lematization se implemento una sola libreria.

###Metodos de aprendizaje automatico utilizados
Para el entrenamiento de los clasificadores, se consideraron dos grandes categorias, la primera con clasificadores que utilicen embeddings y la segunda con clasificadores que utilicen los valores obtenidos del metodo TF-IDF.

#### Metodos con TF-IDF
Con estos metodos simplemente se utilizo la funcion TfidfVectorizer de sklearn, entrenando y realizando el fitting con los datos de entrenmiento y test.

Para el enrenamiento, se probo con cuatros modelos diferentes: Regresion logistica (sklearn), Random Forest (sklearn), Clasificador XBG (XGBoost) y Clasificador LGBMC (lightbgm).
La combinacion entre malos resultados obtenidos y la poca familiaridad con los modelos (en el caso de los ultimos dos), llevo a que se descartara esta linea.

#### Metodos con Embeddings

Para el embedding se implementaron tres metodos diferentes pero todos utilizando fasttext para su realizacion. El primero consiste en cargar los embeddings que fueron entregados para realizar la tarea, el segundo en entrenar un modelo de embeddings en base al corpus de entrenamiento y finalmente importar el modelo preentrenado para el español por fasttext.
Debido a que el modelo entrenado con el corpus de entrenamiento no tenia suficientes cantidades de texto, la calidad de los embeddings no era suficiente para dar buenos resultados. Y en el caso del embedding importado de fasttext, debido al peso del mismo, el tiempo que agregaba a las ejecuciones hacia imposible que se ejecutara dentro del plazo exigido.
Es por esto que se termino utilizando el embedding implementado originalmente, a pesar de que los mejores resultados se obtuvieron con la tercer implementacion.



### Posibles mejoras
