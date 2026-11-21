"""Compuerta de calidad: valida que el modelo entrenado cumpla un umbral mínimo.

Lee los resultados desde MLflow (sqlite:///mlflow.db) y falla (código 1) si:
  - la última ejecución 'Baseline' tiene accuracy < UMBRAL, o
  - el mejor cv_accuracy_mean de los experimentos es < UMBRAL.
"""
import sys

import mlflow

EXPERIMENTO = "Proyecto_Final_Grado"
UMBRAL = 0.90

mlflow.set_tracking_uri("sqlite:///mlflow.db")
exp = mlflow.get_experiment_by_name(EXPERIMENTO)
if exp is None:
    print(f"[FALLO] No existe el experimento '{EXPERIMENTO}'. Ejecuta primero el entrenamiento.")
    sys.exit(1)

runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
ok = True

base = runs[(runs["tags.mlflow.runName"] == "Baseline") & (runs["status"] == "FINISHED")]
if base.empty:
    print("[FALLO] No hay ejecuciones 'Baseline' finalizadas.")
    ok = False
else:
    acc = base.sort_values("start_time").iloc[-1]["metrics.accuracy"]
    estado = "OK" if acc >= UMBRAL else "FALLO"
    ok = ok and acc >= UMBRAL
    print(f"[{estado}] Baseline accuracy = {acc:.4f} (umbral {UMBRAL})")

if "metrics.cv_accuracy_mean" in runs:
    cv = runs["metrics.cv_accuracy_mean"].dropna()
    if cv.empty:
        print("[FALLO] No hay métricas de validación cruzada. Ejecuta experimentos.py.")
        ok = False
    else:
        mejor = cv.max()
        estado = "OK" if mejor >= UMBRAL else "FALLO"
        ok = ok and mejor >= UMBRAL
        print(f"[{estado}] Mejor CV accuracy = {mejor:.4f} (umbral {UMBRAL})")
else:
    print("[FALLO] No hay métricas de validación cruzada. Ejecuta experimentos.py.")
    ok = False

sys.exit(0 if ok else 1)
