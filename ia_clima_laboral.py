# ia_clima_api.py
import pandas as pd
import pymysql
import datetime
import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline

load_dotenv()

app = FastAPI()

# Configurar CORS para React (ajusta el origen si quieres seguridad)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cambia por el dominio de tu frontend en prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def cargar_datos_mysql():
    try:
        print("Intentando conectar a la base de datos...")
        conn = pymysql.connect(
            host=os.environ['DB_HOST'],
            user=os.environ['DB_USER'],
            password=os.environ['DB_PASSWORD'],
            database=os.environ['DB_NAME'],
            port=int(os.environ['DB_PORT'])
        )
        print("Conexión exitosa.")

        empleados = pd.read_sql("SELECT id_empleado, nombre, id_departamento FROM empleado", conn)
        print(f"Cargados empleados: {len(empleados)}")

        departamentos = pd.read_sql("SELECT id_departamento, nombre AS departamento FROM departamento", conn)
        print(f"Cargados departamentos: {len(departamentos)}")

        respuestas = pd.read_sql("""
            SELECT r.id_empleado, p.texto AS pregunta, r.respuesta,
                   CASE WHEN p.tipo IN ('abierta', 'texto') THEN 'texto' ELSE 'numerica' END AS tipo
            FROM respuesta r
            JOIN pregunta p ON r.id_pregunta = p.id_pregunta
        """, conn)
        print(f"Cargadas respuestas: {len(respuestas)}")

        conn.close()
        return empleados, departamentos, respuestas
    except Exception as e:
        print(f"Error al conectar o consultar la BD: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

def analizar_sentimientos(df):
    print("Analizando sentimientos...")
    clasificador = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
    df['sentimiento'] = df['respuesta'].apply(lambda x: clasificador(x[:512])[0]['label'])
    df['polaridad'] = df['sentimiento'].str.extract(r'(\d)').astype(int)
    print("Análisis completado.")
    return df

def guardar_historico(df):
    try:
        print("Guardando histórico de sentimientos...")
        conn = pymysql.connect(
            host=os.environ['DB_HOST'],
            user=os.environ['DB_USER'],
            password=os.environ['DB_PASSWORD'],
            database=os.environ['DB_NAME'],
            port=int(os.environ['DB_PORT'])
        )
        cursor = conn.cursor()
        hoy = datetime.date.today()
        polaridades = df.groupby('departamento')['polaridad'].mean().reset_index()

        for _, row in polaridades.iterrows():
            cursor.execute("""
                INSERT INTO historico_sentimientos (departamento, polaridad_promedio, fecha)
                VALUES (%s, %s, %s)
            """, (row['departamento'], row['polaridad'], hoy))
        conn.commit()
        conn.close()
        print("Histórico guardado exitosamente.")
    except Exception as e:
        print(f"Error al guardar histórico: {e}")

def resumen_automatico(df):
    print("Generando resumen automático...")
    promedios = df.groupby("departamento")["polaridad"].mean()
    peor_dep = promedios.idxmin()
    mejor_dep = promedios.idxmax()
    resumen = {
        "departamento_peor_clima": peor_dep,
        "polaridad_peor": round(promedios[peor_dep], 2),
        "departamento_mejor_clima": mejor_dep,
        "polaridad_mejor": round(promedios[mejor_dep], 2)
    }
    print(f"Resumen generado: {resumen}")
    return resumen

@app.get("/analisis")
def api_analisis():
    empleados, departamentos, respuestas = cargar_datos_mysql()

    # Mejor manejo si alguna tabla está vacía
    if empleados.empty or departamentos.empty or respuestas.empty:
        error_msg = "Las tablas están vacías o no hay datos suficientes."
        print(error_msg)
        return {
            "error": error_msg,
            "cantidad_empleados": len(empleados),
            "cantidad_departamentos": len(departamentos),
            "cantidad_respuestas": len(respuestas)
        }

    respuestas_texto = respuestas[respuestas['tipo'] == 'texto']
    respuestas_completas = respuestas_texto.merge(empleados, on="id_empleado") \
                                           .merge(departamentos, on="id_departamento")

    # Si no hay respuestas abiertas, tampoco caigas
    if respuestas_texto.empty:
        error_msg = "No hay respuestas de tipo texto para analizar."
        print(error_msg)
        return {
            "error": error_msg,
            "cantidad_empleados": len(empleados),
            "cantidad_departamentos": len(departamentos),
            "cantidad_respuestas_texto": 0
        }

    respuestas_con_sentimientos = analizar_sentimientos(respuestas_completas)
    guardar_historico(respuestas_con_sentimientos)
    resumen = resumen_automatico(respuestas_con_sentimientos)

    return {
        "resumen": resumen,
        "cantidad_empleados": len(empleados),
        "cantidad_departamentos": len(departamentos),
        "cantidad_respuestas_texto": len(respuestas_texto)
    }
