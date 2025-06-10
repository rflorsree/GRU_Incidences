import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

# === Preparación ===
output_dir = "outputs"
model_dir = "models"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# Cargar datos
file_path = "data/EXCELINCIDENCIAS.xlsx"
df = pd.read_excel(file_path, sheet_name="Sheet1")

# Procesamiento de fechas
df["INICIO INCIDENCIA"] = pd.to_datetime(df["INICIO INCIDENCIA"])
df["HORA DE LLEGADA"] = pd.to_datetime(df["HORA DE LLEGADA"])
df["CIERRE DE INCIDENCIA"] = pd.to_datetime(df["CIERRE DE INCIDENCIA"])

# Variables temporales
df["hora_inicio"] = df["INICIO INCIDENCIA"].dt.hour
df["mes"] = df["INICIO INCIDENCIA"].dt.month
df["dia_semana"] = df["INICIO INCIDENCIA"].dt.weekday
df["semana_del_anio"] = df["INICIO INCIDENCIA"].dt.isocalendar().week.astype(int)
df["minutos_respuesta"] = (df["HORA DE LLEGADA"] - df["INICIO INCIDENCIA"]).dt.total_seconds() / 60

# Agregar incidencias por día
df["FECHA"] = df["INICIO INCIDENCIA"].dt.date
incidencias_por_dia = df.groupby("FECHA").size().reset_index(name="incidencias")
df = df.merge(incidencias_por_dia, on="FECHA", how="left")

# Variables de entrada y salida
X = df[["hora_inicio", "mes", "dia_semana", "semana_del_anio", "minutos_respuesta"]].values
y = df[["incidencias", "CLIENTES"]].values

# Escalado
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# Guardar escaladores
joblib.dump(scaler_X, os.path.join(model_dir, "scaler_X.pkl"))
joblib.dump(scaler_y, os.path.join(model_dir, "scaler_y.pkl"))

# Guardar gráficos de escalado
plt.figure(figsize=(10, 5))
plt.plot(X_scaled)
plt.title("X - Variables de entrada escaladas")
plt.grid()
plt.savefig(f"{output_dir}/x_scaled_plot.png")
plt.close()

plt.figure(figsize=(10, 5))
plt.plot(y_scaled)
plt.title("y - Variables de salida escaladas")
plt.grid()
plt.savefig(f"{output_dir}/y_scaled_plot.png")
plt.close()

# Redimensionar para GRU
X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

# División de datos
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Comparar configuraciones
configs = [
    {"epochs": 50, "batch_size": 16},
    {"epochs": 100, "batch_size": 32},
    {"epochs": 150, "batch_size": 64}
]

resultados = []
history_best = None
y_pred_best = None
y_test_orig_best = None
labels = ["Incidencias", "Clientes"]

for config in configs:
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_scaled.shape[1], X_scaled.shape[2])),
        tf.keras.layers.GRU(64),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(2)
    ])
    model.compile(optimizer='adam', loss='mse')

    history = model.fit(X_train, y_train, epochs=config["epochs"], batch_size=config["batch_size"], validation_data=(X_test, y_test), verbose=0)

    y_pred = scaler_y.inverse_transform(model.predict(X_test))
    y_test_orig = scaler_y.inverse_transform(y_test)

    mae = mean_absolute_error(y_test_orig, y_pred)
    mse = mean_squared_error(y_test_orig, y_pred)
    r2 = r2_score(y_test_orig, y_pred)

    resultados.append({
        "epochs": config["epochs"],
        "batch_size": config["batch_size"],
        "mae": mae,
        "mse": mse,
        "r2": r2
    })

    if config["epochs"] == 100 and config["batch_size"] == 32:
        history_best = history
        y_pred_best = y_pred
        y_test_orig_best = y_test_orig
        model.save(os.path.join(model_dir, "gru_model.keras"))

# Guardar CSV de comparaciones
df_comp = pd.DataFrame(resultados)
df_comp.to_csv(f"{output_dir}/comparacion_configuraciones.csv", index=False)

# Gráfica doble eje de comparaciones
fig, ax1 = plt.subplots(figsize=(10, 6))
ax2 = ax1.twinx()

x = range(len(resultados))
labels_x = [f"{r['epochs']}/{r['batch_size']}" for r in resultados]

ax1.bar(x, [r["mae"] for r in resultados], width=0.3, label="MAE", align='center')
ax1.bar([i + 0.3 for i in x], [r["mse"] for r in resultados], width=0.3, label="MSE", align='center')
ax2.plot(x, [r["r2"] for r in resultados], color='green', marker='o', label="R²")

ax1.set_xlabel("Configuración (Épocas/Batch Size)")
ax1.set_ylabel("MAE / MSE")
ax2.set_ylabel("R²")

ax1.set_xticks(x)
ax1.set_xticklabels(labels_x)
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
plt.title("Comparación de Configuraciones - GRU")
plt.grid()
plt.savefig(f"{output_dir}/comparacion_configuraciones_doble_eje.png")
plt.close()

# === GRAFICAS ADICIONALES ===
# Pérdida
plt.figure(figsize=(10, 5))
plt.plot(history_best.history['loss'], label='Train Loss')
plt.plot(history_best.history['val_loss'], label='Val Loss')
plt.title('GRU - Pérdida durante el entrenamiento')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.grid()
plt.savefig(f"{output_dir}/gru_loss_curve_2targets.png")
plt.close()

# Real vs Predicho
for i, label in enumerate(labels):
    plt.figure(figsize=(8, 4))
    plt.scatter(y_test_orig_best[:, i], y_pred_best[:, i], alpha=0.5)
    plt.xlabel("Real")
    plt.ylabel("Predicho")
    plt.title(f"GRU - Real vs Predicho - {label}")
    plt.grid()
    fname = f"gru_real_vs_pred_{label.lower().replace(' ', '_')}.png"
    plt.savefig(f"{output_dir}/{fname}")
    plt.close()

# Métricas por variable
maes = [mean_absolute_error(y_test_orig_best[:, i], y_pred_best[:, i]) for i in range(2)]
mses = [mean_squared_error(y_test_orig_best[:, i], y_pred_best[:, i]) for i in range(2)]
r2s = [r2_score(y_test_orig_best[:, i], y_pred_best[:, i]) for i in range(2)]

plt.figure(figsize=(10, 5))
plt.bar(labels, maes, color='purple')
plt.title('GRU - MAE por variable de salida')
plt.ylabel('Mean Absolute Error')
plt.grid(axis='y')
plt.savefig(f"{output_dir}/gru_mae_2targets.png")
plt.close()

plt.figure(figsize=(10, 5))
plt.bar(labels, mses, color='red')
plt.title('GRU - MSE por variable de salida')
plt.ylabel('Mean Squared Error')
plt.grid(axis='y')
plt.savefig(f"{output_dir}/gru_mse_2targets.png")
plt.close()

plt.figure(figsize=(10, 5))
plt.bar(labels, r2s, color='green')
plt.title('GRU - R² Score por variable de salida')
plt.ylabel('R²')
plt.ylim(0, 1)
plt.grid(axis='y')
plt.savefig(f"{output_dir}/gru_r2_2targets.png")
plt.close()
