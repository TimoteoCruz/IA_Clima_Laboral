# -*- coding: utf-8 -*-
"""IA_Clima_Laboral_Mejorado.py"""

import pandas as pd
import pymysql
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline
import datetime
import os
from dotenv import load_dotenv
load_dotenv()

# ✅ Configurar  conexión MySQL
def cargar_datos_mysql():
    try:
        conn = pymysql.connect(
    host=os.environ['DB_HOST'],
    user=os.environ['DB_USER'],
    password=os.environ['DB_PASSWORD'],
    database=os.environ['DB_NAME'],
    port=int(os.environ['DB_PORT'])
)

        print("🔍 Ejecutando consulta empleados...")
        empleados = pd.read_sql("SELECT id_empleado, nombre, id_departamento FROM empleado", conn)
        print("👥 Empleados cargados:", empleados.shape)

        print("🔍 Ejecutando consulta departamentos...")
        departamentos = pd.read_sql("SELECT id_departamento, nombre AS departamento FROM departamento", conn)
        print("🏢 Departamentos cargados:", departamentos.shape)

        print("🔍 Ejecutando consulta respuestas...")
        respuestas = pd.read_sql("""
            SELECT r.id_empleado, p.texto AS pregunta, r.respuesta,
                   CASE
                       WHEN p.tipo IN ('abierta', 'texto') THEN 'texto'
                       ELSE 'numerica'
                   END AS tipo
            FROM respuesta r
            JOIN pregunta p ON r.id_pregunta = p.id_pregunta
        """, conn)
        print("🗨️ Respuestas cargadas:", respuestas.shape)

        conn.close()
        print("✅ Datos cargados correctamente.")
        return empleados, departamentos, respuestas

    except Exception as e:
        print(f"❌ Error al conectar con la base de datos: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

# 🤖 Análisis de sentimientos
def analizar_sentimientos(df):
    print("🧠 Analizando sentimientos de comentarios...")
    clasificador = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

    df['sentimiento'] = df['respuesta'].apply(lambda x: clasificador(x[:512])[0]['label'])
    df['polaridad'] = df['sentimiento'].str.extract(r'(\d)').astype(int)

    print(df[['respuesta', 'sentimiento', 'polaridad']].head())
    return df

# 📊 Graficar resultados
def graficar_resultados(df):
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="departamento", y="polaridad", estimator='mean', errorbar=None,
                palette="viridis", hue="departamento", legend=False)
    plt.title("Promedio de Sentimientos por Departamento")
    plt.xlabel("Departamento")
    plt.ylabel("Polaridad promedio (1 = negativa, 5 = positiva)")
    plt.xticks(rotation=45)
    plt.tight_layout()
plt.savefig("sentimientos_por_departamento.png")
print("📷 Gráfico guardado como 'sentimientos_por_departamento.png'")

# 📦 Guardar en tabla histórica
def guardar_historico(df):
    try:
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
        print("📦 Datos guardados en el histórico.")
    except Exception as e:
        print(f"❌ Error al guardar histórico: {e}")

# 🧾 Resumen ejecutivo automático
def resumen_automatico(df):
    promedios = df.groupby("departamento")["polaridad"].mean()
    peor_dep = promedios.idxmin()
    mejor_dep = promedios.idxmax()

    print("\n📝 RESUMEN EJECUTIVO:")
    print(f"📉 Departamento con menor clima laboral: {peor_dep} ({promedios[peor_dep]:.2f})")
    print(f"📈 Departamento con mejor clima laboral: {mejor_dep} ({promedios[mejor_dep]:.2f})")

# 🧠 Lógica principal
def main():
    print("\n1️⃣ CARGANDO DATOS...")
    empleados, departamentos, respuestas = cargar_datos_mysql()

    if empleados.empty or departamentos.empty or respuestas.empty:
        print("❌ No se pudieron cargar los datos.")
        return

    print("\n2️⃣ FILTRANDO RESPUESTAS ABIERTAS...")
    respuestas_texto = respuestas[respuestas['tipo'] == 'texto']

    respuestas_completas = respuestas_texto.merge(empleados, on="id_empleado") \
                                           .merge(departamentos, on="id_departamento")

    print("\n3️⃣ ANALIZANDO SENTIMIENTOS...")
    respuestas_con_sentimientos = analizar_sentimientos(respuestas_completas)

    print("\n4️⃣ MOSTRANDO RESULTADOS...")
    graficar_resultados(respuestas_con_sentimientos)

    print("\n5️⃣ GUARDANDO HISTÓRICO...")
    guardar_historico(respuestas_con_sentimientos)

    print("\n6️⃣ GENERANDO RESUMEN...")
    resumen_automatico(respuestas_con_sentimientos)

# 🚀 Ejecutar
if __name__ == "__main__":
    main()
