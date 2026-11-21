# Orquestación del pipeline MLOps (funciona en Linux/macOS y Windows con make)
ifeq ($(OS),Windows_NT)
PY_SISTEMA := python
PY := .venv/Scripts/python.exe
else
PY_SISTEMA := python3
PY := .venv/bin/python
endif

.PHONY: help all setup validate-data train experiments validate-model report clean

help:
	@echo "make setup           -> crea .venv e instala dependencias"
	@echo "make all             -> ejecuta TODO el pipeline (validacion + entrenamiento + reporte)"
	@echo "make validate-data   -> valida contrato y dataset"
	@echo "make train           -> entrena el baseline"
	@echo "make experiments     -> corre los experimentos comparativos"
	@echo "make validate-model  -> compuerta de calidad del modelo"
	@echo "make report          -> genera tabla y grafico"
	@echo "make clean           -> elimina mlflow.db, mlruns y reportes generados"

all: validate-data train experiments validate-model report

setup:
	$(PY_SISTEMA) -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt

validate-data:
	$(PY) validar_datos.py

train: validate-data
	$(PY) train.py

experiments: train
	$(PY) experimentos.py

validate-model: experiments
	$(PY) validar_modelo.py

report: validate-model
	$(PY) reporte.py

clean:
	$(PY_SISTEMA) -c "import shutil,glob,os; shutil.rmtree('mlruns', ignore_errors=True); [os.remove(f) for f in glob.glob('mlflow.db')+glob.glob('tabla_comparativa*')+glob.glob('grafico_metricas*')]"
