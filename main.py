from fastapi import FastAPI
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

app = FastAPI()

# Carga los datos parquet en un dataframe de pandas
Tabla_API = pd.read_parquet('Tabla_API.parquet')


@app.get('/PlayTimeGenre')
def PlayTimeGenre(genero: str):
    #Filtrar el DataFrame por el genero especificado
    df_genre = Tabla_API[Tabla_API['genres'].apply(lambda x: genero in x)]

    #Comprobar si df_genre esta vacio
    if df_genre.empty:
        return 'No hay datos para el género especificado.'

    #Encontrar el año con mas tiempo de juego
    year_max_playtime = df_genre.groupby('year')['playtime_forever'].sum().idxmax()

    #Crear un diccionario con los resultados
    return {'Año de lanzamiento con más horas jugadas para Género ' + genero: int(year_max_playtime)}

@app.get('/UserForGenre')
def UserForGenre(genero: str):
    #Filtrar el DataFrame por el genero especificado
    df_genre = Tabla_API[Tabla_API['genres'].apply(lambda x: genero in x)]
    #Comprobar si df_genre está vacío
    if df_genre.empty:
        return 'No hay datos para el género especificado.'

    #Encontrar el usuario que acumula más horas jugadas
    user_max_hours = df_genre.groupby('user_id')['playtime_forever'].sum().idxmax()

    #Calcular la acumulación de horas jugadas por año de lanzamiento
    hours_per_year = df_genre.groupby('year_posted')['playtime_forever'].sum().reset_index().to_dict('records')

    #Crear un diccionario con los resultados
    return ({
        'Usuario con más horas jugadas para Género ' + genero: user_max_hours,
        'Horas jugadas': hours_per_year
    })

@app.get('/UsersRecommend')
def UsersRecommend(year: int):
    #Filtrar el DataFrame por año y recomendaciones positivas
    df_year = Tabla_API[(Tabla_API['year_posted'] == year) & (Tabla_API['recommend'] == True)]

    #Contar el numero de recomendaciones positivas por juego
    df_count = df_year.groupby('title')['recommend'].count()

    #Obtener los 3 mejores juegos con las recomendaciones más positivas
    top_games = df_count.nlargest(3)

    #Crear una lista con los resultados
    return [{'Puesto ' + str(i+1): game} for i, game in enumerate(top_games.index)]

@app.get('/UsersWorstDeveloper')
def UsersWorstDeveloper(year: int):
    #Filtar el DataFrame por año espeficado y recomendaciones negativas
    df_year = Tabla_API[(Tabla_API['year_posted'] == year) & (Tabla_API['recommend'] == False)]

    #Contar el numero de recomendaciones negativas por desarrollador
    df_count = df_year.groupby('developer')['recommend'].count()

    #Obtener los 3 juegos con las recomendaciones más negativas
    worst_developers = df_count.nlargest(3)

    #Crear una lista con los resultados
    return [{'Puesto ' + str(i+1): developer} for i, developer in enumerate(worst_developers.index)]

@app.get('/sentiment_analysis')
def sentiment_analysis(developer: str):
    # Filtrar el DataFrame por desarrollador especificado
    df_developer = Tabla_API[Tabla_API['developer'] == developer]

    #Comprobar si df_developer esta vacio
    if df_developer.empty:
        return 'No hay datos para el desarrollador especificado.'

    #Contar el numero de cada sentimiento
    sentiment_counts = df_developer['review'].value_counts().to_dict()

    #Crear un diccionario con los resultados
    return [{developer: {'Negative': sentiment_counts.get(0, 0), 'Neutral': sentiment_counts.get(1, 0), 'Positive': sentiment_counts.get(2, 0)}}]


def recomendacion_juego(item_id):
    #Cargar datos para utilizar en dos Dataframes distintos
    df = pd.read_csv('DatasetsLimpios\\games_steam.csv')
    df1 = pd.read_csv('DatasetsLimpios\\games_id.csv')

    #Crear una matriz de juegos
    tfidv = TfidfVectorizer(min_df=2, max_df=0.7, token_pattern=r'\b[a-zA-Z0-9]\w+\b')
    data_vector = tfidv.fit_transform(df['features'])

    data_vector_df = pd.DataFrame(data_vector.toarray(), index=df['item_id'], columns = tfidv.get_feature_names_out())

    #Calcular la similitud del coseno entre los juegos en la matriz de juegos
    vector_similitud_coseno = cosine_similarity(data_vector_df.values)

    cos_sim_df = pd.DataFrame(vector_similitud_coseno, index=data_vector_df.index, columns=data_vector_df.index)

    juego_simil = cos_sim_df.loc[item_id]

    simil_ordenada = juego_simil.sort_values(ascending=False)
    resultado = simil_ordenada.head(6).reset_index()

    result_df = resultado.merge(df1, on='item_id',how='left')

    #Devolver una lista con los 6 juegos mas similares
    juego_title = df1[df1['item_id'] == item_id]['title'].values[0]

    #Devuelve un mensaje que indica el juego original y los juegos recomendados
    mensaje = f"Si te gustó el juego {item_id} : {juego_title}, también te pueden gustar:"

    result_dict = {
        'mensaje': mensaje,
        'juegos recomendados': result_df['title'][1:6].tolist()
    }

    return result_dict