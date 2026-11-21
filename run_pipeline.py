"""Orquestador del pipeline: entrenamiento + validación con UN solo comando.

  python run_pipeline.py --setup   # primera vez: crea .venv, instala dependencias y ejecuta todo
  python run_pipeline.py           # ejecuta el pipeline (usa .venv si existe)

Pasos (se detiene en el primer error):
  1. Validación de datos        -> validar_datos.py
  2. Entrenamiento baseline     -> train.py
  3. Experimentos comparativos  -> experimentos.py
  4. Validación del modelo      -> validar_modelo.py
  5. Reporte (tabla + gráfico)  -> reporte.py
"""
import subprocess
import sys
import time
import venv
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
VENV = RAIZ / ".venv"
PY_VENV = VENV / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")

PASOS = [
    ("Validación de datos", "validar_datos.py"),
    ("Entrenamiento (baseline)", "train.py"),
    ("Experimentos comparativos", "experimentos.py"),
    ("Validación del modelo", "validar_modelo.py"),
    ("Reporte final", "reporte.py"),
]


def ejecutar(cmd):
    return subprocess.run([str(c) for c in cmd], cwd=RAIZ).returncode


def preparar_entorno():
    if not PY_VENV.exists():
        print(">> Creando entorno virtual en .venv ...")
        venv.create(VENV, with_pip=True)
    print(">> Instalando dependencias (requirements.txt) ...")
    if ejecutar([PY_VENV, "-m", "pip", "install", "--upgrade", "pip"]) != 0:
        sys.exit("No se pudo actualizar pip.")
    if ejecutar([PY_VENV, "-m", "pip", "install", "-r", "requirements.txt"]) != 0:
        sys.exit("No se pudieron instalar las dependencias.")


def main():
    if "--setup" in sys.argv:
        preparar_entorno()

    python = PY_VENV if PY_VENV.exists() else Path(sys.executable)
    print(f">> Intérprete: {python}\n")

    inicio = time.time()
    for i, (titulo, script) in enumerate(PASOS, 1):
        print(f"===== [{i}/{len(PASOS)}] {titulo} ({script}) =====")
        if ejecutar([python, script]) != 0:
            print(f"\n[ERROR] Falló el paso {i}: {titulo}. Pipeline detenido.")
            sys.exit(1)
        print()

    print(f"Pipeline completado en {time.time() - inicio:.1f} s")
    print("Resultados: tabla_comparativa_final.csv/.xlsx, grafico_metricas_final.png, mlflow.db")
    print("Para ver la interfaz de MLflow:  mlflow ui --backend-store-uri sqlite:///mlflow.db")


if __name__ == "__main__":
    main()
