# ProyectoAPI

# <h1 align=center> **PROYECTO INDIVIDUAL Nº1**
# <h1 align=center> **Andres Felipe Corzo**
### <h1 align=center> `Machine Learning Operations` (MLOps)

Este proyecto se enfocó en resolver un desafío de Steam: crear un sistema de recomendación de videojuegos para usuarios. Durante su desarrollo, se realizaron tareas de Data Engineer para construir un MVP (Minimum Viable Product) centrado en consultar datos específicos y ofrecer recomendaciones de juegos basadas en las elecciones de los usuarios en la plataforma Steam.

Enlaces útiles:
- Repositorio en GitHub: [AndresFCorzo/ProyectoAPI](https://github.com/AndresFCorzo/ProyectoAPI)
- Despliegue: [proyectoapi-zrul.onrender/docs](https://proyectoapi-zrul.onrender.com/docs)
- Video explicativo: 

El objetivo principal fue crear una API utilizando FastAPI para acceder a datos de manera específica y garantizar su calidad después de un proceso exhaustivo de ETL.

## <h1 align=center> **`Extracción, Transformación y Carga de Datos (Descripción General)`**

(Documentos: ETL_steam_games.ipynb, ETL_user_items.ipynb, ETL_user_reviews.ipynb)

Se trabajó en la lectura y procesamiento de 3 archivos JSON para hacerlos manipulables en Python. Se convirtieron en diccionarios y luego en DataFrames de pandas para su análisis.

Detalles de limpieza, transformación y condiciones específicas para cada dataset se incluyen en esta sección.

    1. `explode()`: Esta función se emplea para desglosar una lista en elementos individuales y crear filas separadas para cada uno de ellos. En este contexto, se utiliza para expandir una columna con datos anidados en un DataFrame, generando nuevas filas según los valores presentes en esa columna.

    2. `json_normalize()`: Utilizada para convertir datos JSON semi-estructurados en una tabla plana, esta función transforma una columna específica con datos anidados (una lista de diccionarios) en un DataFrame. 

    3. `concat()`: Es utilizada para unir dos o más DataFrames a lo largo de un eje específico, generalmente columnas. En este caso, se emplea para combinar el DataFrame original con el nuevo DataFrame generado a partir de datos anidados presentes en una columna.

    4. `drop()`: Una función que permite eliminar etiquetas específicas de filas o columnas en un DataFrame. En este contexto, se usó para eliminar columnas con poca o nula relevancia para el análisis, contribuyendo así a simplificar el conjunto de datos.

    5. `dropna()`: Utilizada para eliminar filas que contienen valores faltantes en las columnas seleccionadas. Esto ayuda a limpiar el DataFrame al eliminar registros con información incompleta.

    7. `drop_duplicates()`: Empleada para eliminar filas duplicadas dentro del DataFrame, manteniendo únicamente registros únicos y eliminando la redundancia en los datos.

    8. Guardar archivos: La funcion `to_parquet` se usa para escribir un archivo en formato parquet. En este caso, se utiliza para guardar el Dataframe en 3 archivos llamados: 'user_reviews.parquet', 'user_items.parquet' y 'games.parquet'.

### Condiciones Específicas de Tratamiento de Datos por Dataset:

#### a. Dataset ‘australian_user_items’ - Documento: ETL_user_items.ipynb

Se filtró el DataFrame para considerar solo aquellos ítems ('playtime_forever') jugados durante al menos una hora (diferentes de 0), así `data_users_items = data_users_items[data_users_items['playtime_forever'] != 0]`

#### b. Dataset ‘australian_user_reviews’ - Documento: ETL_user_reviews.ipynb

- Se convirtió la columna 'reviews', una lista de diccionarios, en un DataFrame mediante la función `apply()` y luego se concatenó con el DataFrame original.
- Se crea el analisis de sentimiento  partir de la librería nltk, especificamente con `SentimentIntensityAnalizer()`.
- Se creó una nueva columna 'year_posted' extrayendo el año de la columna 'posted' usando una expresión regular, así `data_user_reviews['year_posted'] = data_user_reviews['posted'].str.extract(r'(\d{4})')`.

**Nota:** Se analizaron únicamente los registros que estuvieron comentados, valorados y recomendados.

#### c. Dataset ‘output_steam_games’ - Documento: ETL_games.ipynb

- Se creó una nueva columna 'year' extrayendo el año de la columna 'release_date' usando la siguiente expresión `data_games['year'] = data_games['release_date'].str.extract(r'(\d{4})')`
- Se cambian el tipo variables de las columnas 'year' y 'developer', así: `data_games.loc['year'] = data_games['year'].astype(int)`, `data_games.loc['developer'] = data_games['developer'].astype(str)`.
- Y se eliminan las ultimas dos filas que aparecen nuevas con valores "Nulos", así: `data_games = data_games.iloc[:-2]`.

### Relación y Fusión de Tablas - Documento: MERGE_API.ipynb

- Se creó un identificador único ('id') en los DataFrames 'items' y 'reviews' al concatenar las columnas 'user_id' y 'item_id'.
- Se fusionaron los DataFrames 'reviews' y 'games' mediante la función `merge()` utilizando la columna 'item_id'.
- Se fusionaron los DataFrames 'items' y 'merged_df' usando 'id'.
- Se seleccionaron columnas específicas en el DataFrame 'steam'.
- Se guardó el DataFrame 'steam' en un archivo .parquet llamado 'Tabla_API.parquet' mediante la función `to_parquet()`.


## <h1 align=center> **`Análisis de Datos Exploratorio`** 

(Documento: EDA.ipynb)

Se exploraron datos, se trató la presencia de valores atípicos y se analizaron variables clave para comprender mejor el conjunto de datos.

En el análisis estadístico y gráfico de las variables numéricas, se identificaron outliers en 'playtime_forever'. Estos valores atípicos no fueron eliminados debido a que su cantidad corresponde al casi 100% de los datos que contiene el merge realizado.

El cálculo del Rango Intercuartil (IQR) se llevó a cabo en la columna 'paytime_forever'. Para ello, se utilizó `Q1 = data.quantile(0.25)` para determinar el primer cuartil y `Q3 = data.quantile(0.75)` para el tercer cuartil. La diferencia entre estos cuartiles, `IQR = Q3 - Q1`, proporciona una medida robusta de dispersión estadística, abarcando el 50% central de los datos.

Es importante destacar que el IQR es menos sensible a los valores atípicos en comparación con el rango, convirtiéndose así en una medida más robusta de variabilidad.

Después de ello se realiza la correlacion de las variables mediante la graficación de las variables anteriores, utlizando `sns.pairplot(df), plt.show()` y una matriz de correlacion así: `df_numeric = df.select_dtypes(include=['int64']), corr = df_numeric.corr(), sns.heatmap(corr, annot=True, cmap='coolwarm'), plt.show()`


## <h1 align=center> **`Machine Learning`**

(Documento: MachineLearning.ipynb)

Preparación de datos para un modelo de recomendación basado en similitud de coseno y explicación detallada del código utilizado.

**Preparación de datos:**
Para la preparación de datos con el método de Similitud de Coseno y las variables más relacionadas con el objetivo, se generaron dos subconjuntos de datos. Por un lado, 'games_id' incluye 'item_id' y 'título del juego', mientras que el otro, 'games_steam', contiene 'item_id' y 'features'. Esta última columna, 'features', representa la combinación de 'title', 'developer' y 'release_date' en una única columna con valores separados por comas y espacios. Estas variables se seleccionaron específicamente para aplicar el modelo de similitud de coseno en la recomendación.

**Modelo de recomendación:**
El código de la función `recomendacion_juego(item_id)` se descompone paso a paso:

- `df = pd.read_csv('DatasetsLimpios\\games_steam.csv')` y `df1 = pd.read_csv('DatasetsLimpios\\games_id.csv')` leen datos de dos archivos CSV en DataFrames pandas.

- `tfidv = TfidfVectorizer(min_df=2, max_df=0.7, token_pattern=r'\b[a-zA-Z0-9]\w+\b')` inicializa un TfidfVectorizer, convirtiendo el texto en vectores de características. Se configuran parámetros como frecuencia mínima y máxima de documentos para filtrar términos.

- `data_vector = tfidv.fit_transform(data['features'])` ajusta y transforma la columna 'features' del DataFrame 'data' en una matriz de TF-IDF.

- `data_vector_df = pd.DataFrame(data_vector.toarray(), index=data['item_id'], columns=tfidv.get_feature_names_out())` convierte la matriz TF-IDF en un DataFrame.

- `vector_similitud_coseno = cosine_similarity(data_vector_df.values)` calcula la similitud del coseno entre pares de vectores TF-IDF.

- `cos_sim_df = pd.DataFrame(vector_similitud_coseno, index=data_vector_df.index, columns=data_vector_df.index)` convierte la matriz de similitud de coseno en un DataFrame.

- `juego_simil = cos_sim_df.loc[item_id]` selecciona la fila correspondiente al 'item_id' del juego de entrada del DataFrame de similitud de coseno.

- `simil_ordenada = juego_simil.sort_values(ascending=False)` ordena la fila seleccionada en orden descendente de similitud de coseno.

- `resultado = simil_ordenada.head(6).reset_index()` elige los 6 juegos más similares al juego de entrada y restablece el índice del DataFrame resultante.

- `result_df = resultado.merge(data_juegos_steam, on='item_id', how='left')` fusiona el DataFrame resultante con 'data_juegos_steam' según 'item_id' para obtener los títulos de los juegos recomendados.

Se generan mensajes recomendando los 5 mejores juegos similares al de entrada y se almacenan en un diccionario.

`return result_dict` devuelve el diccionario con el mensaje y los juegos recomendados.


## <h1 align=center> **`Funciones y API`**

(Documento: main.py)

Este apartado detalla los endpoints disponibles en la API, cada uno realizando diferentes análisis de datos a partir del conjunto disponible, facilitando consultas específicas.

Los endpoints incluyen:

`'/PlayTimeGenre'`: Este endponit devuelve el año con mas horas jugadas para un genero en especifico, el cual es introducido.

`'/UserForGenre'`: Este endpoint devuelve el usuario que ha pasado más horas jugando para un género determinado y una lista de la acumulación de horas jugadas por año.

`'/UsersRecommend'`: Este endpoint devuelve un top de 3 juegos con mas recomendaciones por usuarios para un año en especifico el cual es introducido.

`'/UsersWorstDeveloper'`: Este endpoint devuelve un top de 3 desarrolladores con juegos menos recomendados para un año dado que es introducido.

`'/sentiment_analysis'`: Este endpoint devuelve devuelve un diccionario con el nombre de la desarrolladora como llave y una lista con la cantidad total de registros de reseñas de usuarios que se encuentren categorizados con un análisis de sentimiento como valor, según la desarrolladora la cual es introducida.

`'/recomendacion_juego'`: Este endpoint es el modelo de Machine Learning el cual ingresando el id de producto (item_id), deberíamos recibir una lista con 5 juegos recomendados similares al ingresado.

El estado actual del proyecto es: completo/publicado.