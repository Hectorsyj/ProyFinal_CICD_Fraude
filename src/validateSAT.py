# src/validate.py
import pickle
import numpy as np
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
import pandas as pd
import mlflow
import pickle

with open('data/X_test.pkl', 'rb') as f:
    X_test = pickle.load(f)

with open('data/y_test.pkl', 'rb') as f:
    y_test = pickle.load(f)

with open('models/model.pkl', 'wb') as f:
    model = pickle.load(f)

print(f"✅ Modelo y datos cargados correctamente")
print(f"   X_test shape: {X_test.shape}")
print(f"   y_test shape: {y_test.shape}")

try:
    with open("mlflow_run_id.txt", "r") as f:
        run_id = f.read().strip()
except FileNotFoundError:
    print("Error: No se encontró 'mlflow_run_id.txt'. El entrenamiento falló o no guardó el ID.")
    exit(1)

# HACER PREDICCIONES
print("\n🔮 Realizando predicciones...")
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# CALCULAR MÉTRICAS
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

print("\n" + "="*50)
print("📊 RESULTADOS DE VALIDACIÓN")
print("="*50)
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")

print("\n📋 Reporte de clasificación:")
print(classification_report(y_test, y_pred, target_names=['No Fraude', 'Fraude']))

print("\n🔢 Matriz de confusión:")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(f"\nVerdaderos Negativos: {cm[0,0]}")
print(f"Falsos Positivos:     {cm[0,1]}")
print(f"Falsos Negativos:     {cm[1,0]}")
print(f"Verdaderos Positivos: {cm[1,1]}")

# VALIDAR SI EL MODELO CUMPLE CON UMBRALES MÍNIMOS
print("\n" + "="*50)
print("✅ VALIDACIÓN DE UMBRALES")
print("="*50)

umbral_roc_auc = 0.85
umbral_recall = 0.70

if roc_auc >= umbral_roc_auc and recall >= umbral_recall:
    print(f"✅ MODELO APROBADO")
    print(f"   ROC-AUC: {roc_auc:.4f} >= {umbral_roc_auc}")
    print(f"   Recall:  {recall:.4f} >= {umbral_recall}")
    
    # Guardar métricas
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc
    }
    
    with open('models/metrics.pkl', 'wb') as f:
        pickle.dump(metrics, f)
    
    print("\n💾 Métricas guardadas en: models/metrics.pkl")
    exit(0)  # Éxito
else:
    print(f"❌ MODELO RECHAZADO")
    if roc_auc < umbral_roc_auc:
        print(f"   ROC-AUC: {roc_auc:.4f} < {umbral_roc_auc} ❌")
    if recall < umbral_recall:
        print(f"   Recall:  {recall:.4f} < {umbral_recall} ❌")
    exit(1)  # Fallo
