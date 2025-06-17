
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Crear carpeta de salida
output_dir = "eda_outputs"
os.makedirs(output_dir, exist_ok=True)

# Cargar el archivo
df = pd.read_excel("data/EXCELINCIDENCIAS.xlsx")

# Procesar fechas
df["INICIO INCIDENCIA"] = pd.to_datetime(df["INICIO INCIDENCIA"])
df["HORA DE LLEGADA"] = pd.to_datetime(df["HORA DE LLEGADA"])
df["CIERRE DE INCIDENCIA"] = pd.to_datetime(df["CIERRE DE INCIDENCIA"])

# Variables temporales
df["hora_inicio"] = df["INICIO INCIDENCIA"].dt.hour
df["mes"] = df["INICIO INCIDENCIA"].dt.month
df["dia_semana"] = df["INICIO INCIDENCIA"].dt.weekday
df["minutos_respuesta"] = (df["HORA DE LLEGADA"] - df["INICIO INCIDENCIA"]).dt.total_seconds() / 60
df["minutos_resolucion"] = (df["CIERRE DE INCIDENCIA"] - df["INICIO INCIDENCIA"]).dt.total_seconds() / 60

# Guardar resumen estadístico
df.describe(include="all").to_csv(os.path.join(output_dir, "resumen_estadistico.csv"))

# Correlaciones numéricas
corr = df.select_dtypes(include=["float64", "int64"]).corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Matriz de Correlación Numérica")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "matriz_correlacion.png"))
plt.close()

# Boxplot por causa
plt.figure(figsize=(12, 6))
sns.boxplot(x="CAUSA", y="minutos_resolucion", data=df)
plt.xticks(rotation=45)
plt.title("Boxplot de Tiempo de Resolución por Causa")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "boxplot_causa_resolucion.png"))
plt.close()

# Distribución de tiempo de resolución
plt.figure(figsize=(8, 5))
sns.histplot(df["minutos_resolucion"], bins=50, kde=True)
plt.title("Distribución del Tiempo de Resolución")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "histograma_tiempo_resolucion.png"))
plt.close()
