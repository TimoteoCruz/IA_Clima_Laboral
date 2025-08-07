# ia_clima_api.py
import pandas as pd
import psycopg2
import datetime
import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline
from sqlalchemy import create_engine
import os

load_dotenv()

app = FastAPI()

# Configurar CORS para React (ajusta el origen si quieres seguridad)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cambia por tu frontend en producción
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Umbrales para alertas
UMBRAL_BAJO = 3.5
UMBRAL_ALTO = 4.5

def cargar_datos_postgres():
    try:
        print("Intentando conectar a la base de datos con SQLAlchemy...")
        engine = create_engine(
            f"postgresql://{os.environ['DB_USER']}:{os.environ['DB_PASSWORD']}@{os.environ['DB_HOST']}:{os.environ['DB_PORT']}/{os.environ['DB_NAME']}"
        )

        empleados = pd.read_sql("""
            SELECT e.id_empleado, e.nombre, e.id_departamento, e.id_empresa,
                   d.nombre AS departamento, em.nombre AS empresa
            FROM empleado e
            JOIN departamento d ON e.id_departamento = d.id_departamento
            JOIN empresa em ON e.id_empresa = em.id_empresa
        """, engine)
        print(f"Cargados empleados: {len(empleados)}")

        respuestas = pd.read_sql("""
            SELECT r.id_empleado, p.texto AS pregunta, r.respuesta,
                   CASE WHEN p.tipo IN ('abierta', 'texto') THEN 'texto' ELSE 'numerica' END AS tipo
            FROM respuesta r
            JOIN pregunta p ON r.id_pregunta = p.id_pregunta
        """, engine)
        print(f"Cargadas respuestas: {len(respuestas)}")

        return empleados, respuestas

    except Exception as e:
        print(f"Error al conectar o consultar la BD: {e}")
        return pd.DataFrame(), pd.DataFrame()

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
        conn = psycopg2.connect(
            host=os.environ['DB_HOST'],
            user=os.environ['DB_USER'],
            password=os.environ['DB_PASSWORD'],
            dbname=os.environ['DB_NAME'],
            port=int(os.environ['DB_PORT'])
        )
        cursor = conn.cursor()
        hoy = datetime.date.today()
        polaridades = df.groupby(['empresa', 'departamento'])['polaridad'].mean().reset_index()

        for _, row in polaridades.iterrows():
            cursor.execute("""
                INSERT INTO historico_sentimientos (empresa, departamento, polaridad_promedio, fecha)
                VALUES (%s, %s, %s, %s)
            """, (row['empresa'], row['departamento'], row['polaridad'], hoy))
        conn.commit()
        conn.close()
        print("Histórico guardado exitosamente.")
    except Exception as e:
        print(f"Error al guardar histórico: {e}")

def generar_alertas(df):
    alertas = []
    promedios = df.groupby(['empresa', 'departamento'])['polaridad'].mean()

    for (empresa, departamento), valor in promedios.items():
        if valor < UMBRAL_BAJO:
            alertas.append({
                "empresa": empresa,
                "departamento": departamento,
                "nivel": "bajo",
                "mensaje": f"Alerta: clima laboral bajo ({valor:.2f}) en {departamento} de {empresa}."
            })
        elif valor > UMBRAL_ALTO:
            alertas.append({
                "empresa": empresa,
                "departamento": departamento,
                "nivel": "alto",
                "mensaje": f"Buen clima laboral ({valor:.2f}) en {departamento} de {empresa}."
            })
    return alertas

def resumen_automatico(df):
    print("Generando resumen automático...")
    promedios = df.groupby(["empresa", "departamento"])["polaridad"].mean()
    peor = promedios.idxmin()
    mejor = promedios.idxmax()
    resumen = {
        "empresa_peor_clima": peor[0],
        "departamento_peor_clima": peor[1],
        "polaridad_peor": round(promedios[peor], 2),
        "empresa_mejor_clima": mejor[0],
        "departamento_mejor_clima": mejor[1],
        "polaridad_mejor": round(promedios[mejor], 2)
    }
    print(f"Resumen generado: {resumen}")
    return resumen

@app.get("/analisis")
def api_analisis():
    empleados, respuestas = cargar_datos_postgres()

    # Manejo si alguna tabla está vacía
    if empleados.empty or respuestas.empty:
        error_msg = "Las tablas están vacías o no hay datos suficientes."
        print(error_msg)
        return {
            "error": error_msg,
            "cantidad_empleados": len(empleados),
            "cantidad_respuestas": len(respuestas)
        }

    respuestas_texto = respuestas[respuestas['tipo'] == 'texto']
    respuestas_completas = respuestas_texto.merge(empleados, on="id_empleado")

    if respuestas_texto.empty:
        error_msg = "No hay respuestas de tipo texto para analizar."
        print(error_msg)
        return {
            "error": error_msg,
            "cantidad_empleados": len(empleados),
            "cantidad_respuestas_texto": 0
        }

    respuestas_con_sentimientos = analizar_sentimientos(respuestas_completas)
    guardar_historico(respuestas_con_sentimientos)
    resumen = resumen_automatico(respuestas_con_sentimientos)
    alertas = generar_alertas(respuestas_con_sentimientos)

    return {
        "resumen": resumen,
        "alertas": alertas,
        "cantidad_empleados": len(empleados),
        "cantidad_respuestas_texto": len(respuestas_texto)
    }
