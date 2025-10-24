
import mlflow
import os
import shutil
import sys
import traceback


RUN_ID = os.environ.get("MLFLOW_RUN_ID")
print(f'esta es la id del  Run de Wordflow {RUN_ID}')

TEMP_DOWNLOAD_PATH = os.environ.get("DOWNLOAD_PATH", "./temp_model_artifact")  # Debe coincidir con el 'path' del YAML

if not RUN_ID:
    print("Error: MLFLOW_RUN_ID no está definido. ¿Falló el paso 'get_run_id'?")
    sys.exit(1)

# 1. Configurar Tracking URI
# (Asegúrate de que este script pueda acceder al directorio 'mlruns')
mlruns_dir = os.path.join(os.getcwd(), "mlruns")
tracking_uri = "file://" + os.path.abspath(mlruns_dir)
mlflow.set_tracking_uri(tracking_uri)

# 2. 'model' es el artifact_path que usaste en log_model en train.py
model_source_uri = f"runs:/{RUN_ID}/model" 
print(f"Descargando modelo de: {model_source_uri} a {TEMP_DOWNLOAD_PATH}")

# 3. Descargar el artefacto
try:
    # Asegurar que el directorio de descarga esté limpio
    if os.path.exists(TEMP_DOWNLOAD_PATH):
        print(f"Limpiando directorio existente: {TEMP_DOWNLOAD_PATH}")
        shutil.rmtree(TEMP_DOWNLOAD_PATH)
    os.makedirs(TEMP_DOWNLOAD_PATH)

    # Esta función de MLFlow copia el artefacto del run al path local.
    mlflow.artifacts.download_artifacts(
        artifact_uri=model_source_uri,
        dst_path=TEMP_DOWNLOAD_PATH
    )
    print(f"✅ Descarga completada. Archivos disponibles en {TEMP_DOWNLOAD_PATH}")
    
except Exception as e:
    print(f"🚨 ERROR al descargar artefacto de MLFlow: {e}")
    traceback.print_exc()
    sys.exit(1)