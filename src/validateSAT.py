# src/validate.py
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import sys
import matplotlib.pyplot as plt
import mlflow
import os

with open('data/X_scaled.pkl', 'rb') as f:
    X_scaled = pickle.load(f)

with open('data/y.pkl', 'rb') as f:
    y = pickle.load(f)

# 9. Evaluar el modelo
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy:.4f} | Test Loss: {loss:.4f}')

# Parámetro de umbral
acuracVal = 97.0  # Ajusta este umbral según el MSE esperado para load_diabetes

# 9. Evaluar el modelo
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy:.4f} | Test Loss: {loss:.4f}')

# 10. Visualizar desempeño
plt.figure(figsize=(12,5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Entrenamiento')
plt.plot(history.history['val_loss'], label='Validación')
plt.title('Curva de Pérdida')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Validación')
plt.title('Curva de Precisión')
plt.xlabel('Época')
plt.ylabel('Precisión')
plt.legend()

plt.tight_layout()


# División de datos (usar los mismos datos que en entrenamiento no es ideal para validación real,
# pero necesario aquí para que las dimensiones coincidan. Idealmente, tendrías un split dedicado
# o usarías el X_test guardado del entrenamiento si fuera posible)
# Para este ejemplo, simplemente re-dividimos para obtener un X_test con 10 features.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Añadir random_state para consistencia si es necesario
print(f"--- Debug: Dimensiones de X_test: {X_test.shape} ---")  # Debería ser (n_samples, 10)

# --- Cargar modelo previamente entrenado ---
#model_filename = "model.pkl"
#model_path = os.path.abspath(os.path.join(os.getcwd(), model_filename))

#________________________________________________________________
# 1. Configurar MLFlow Tracking (debe apuntar al mismo lugar que el entrenamiento)
# Esto es crucial para que pueda encontrar el run ID
mlruns_dir = os.path.join(os.getcwd(), "mlruns")
tracking_uri = "file://" + os.path.abspath(mlruns_dir)
mlflow.set_tracking_uri(tracking_uri)

# 2. Obtener el Run ID del archivo generado por 'train.py'
try:
    with open("mlflow_run_id.txt", "r") as f:
        run_id = f.read().strip()
except FileNotFoundError:
    print("Error: No se encontró 'mlflow_run_id.txt'. El entrenamiento falló o no guardó el ID.")
    exit(1)

# 3. Construir la URI de carga y Cargar el modelo
# 'model' es el artifact_path que usaste en log_model (artifact_path="model")
model_uri = f"runs:/{run_id}/model"

print(f"--- Debug: Cargando modelo desde URI: {model_uri} ---")

try:
    # se carga el modelo ya entrenado
    modelsk = mlflow.sklearn.load_model(model_uri) 
    
    print("✅ Modelo cargado correctamente.")
#_____________________________________________________ hasta aca se asegura la ruta y la carga del modelo

#print(f"--- Debug: Intentando cargar modelo desde: {model_path} ---")
#try:
#    model = joblib.load(model_path)
except FileNotFoundError:
    print(f"--- ERROR: No se encontró el archivo del modelo en '{model_uri}'. Asegúrate de que el paso 'make train' lo haya guardado correctamente en la raíz del proyecto. ---")
    # Listar archivos en el directorio actual para depuración
    print(f"--- Debug: Archivos en {os.getcwd()}: ---")
    try:
        print(os.listdir(os.getcwd()))
    except Exception as list_err:
        print(f"(No se pudo listar el directorio: {list_err})")
    print("---")
    sys.exit(1)  # Salir con error

# --- Predicción y Validación ---
print("--- Debug: Realizando predicciones ---")
try:
    y_pred = modelsk.predict(X_test)  # Ahora X_test tiene 10 features
except ValueError as pred_err:
    print(f"--- ERROR durante la predicción: {pred_err} ---")
    # Imprimir información de características si el error persiste
    print(f"Modelo esperaba {modelsk.n_features_in_} features.")
    print(f"X_test tiene {X_test.shape[1]} features.")
    sys.exit(1)

mse = mean_squared_error(y_test, y_pred)
print(f"🔍 MSE del modelo: {mse:.4f} (umbral: {THRESHOLD})")

# Validación
if mse <= THRESHOLD:
    print("✅ El modelo cumple los criterios de calidad.")
    sys.exit(0)  # éxito
else:
    print("❌ El modelo no cumple el umbral. Deteniendo pipeline.")
    sys.exit(1)  # error