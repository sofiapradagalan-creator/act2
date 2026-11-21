# Pipeline MLOps – Random Forest sobre Iris

Proyecto de entrenamiento y validación de modelos `RandomForestClassifier`
con seguimiento de experimentos en **MLflow** (backend SQLite). Incluye automatización
con un solo comando, validación de datos sin librerías externas y un **contrato de datos**
para la futura inferencia.

## 1. Estructura del proyecto

| Archivo | Descripción |
|---|---|
| `run_pipeline.py` | Orquestador multiplataforma: ejecuta todo el pipeline con un comando |
| `Makefile` | Alternativa con `make` (targets `setup`, `all`, `train`, `clean`, ...) |
| `validar_datos.py` | Validación de tipos, nulos y rangos (solo biblioteca estándar) |
| `contrato_datos.json` | Contrato de datos oficial (esquema, ejemplo válido e inválido) |
| `train.py` | Entrenamiento del modelo baseline y registro en MLflow |
| `experimentos.py` | 8 configuraciones de hiperparámetros con validación cruzada |
| `validar_modelo.py` | Compuerta de calidad (umbral de accuracy = 0.90) |
| `reporte.py` | Genera tabla comparativa (CSV/XLSX) y gráfico (PNG) |
| `requirements.txt` | Dependencias con versiones fijadas |

## 2. Requisitos previos

- **Python 3.10 o superior** (verifica con `python --version`; en Linux/macOS, `python3 --version`).
- Conexión a internet (solo para instalar dependencias la primera vez).
- *Opcional:* `make` (si se prefiere el Makefile).

No se necesita ningún entorno virtual previo: el pipeline lo crea.

## 3. Reproducción desde cero

```bash
# 1. Obtener el proyecto
git clone https://github.com/sofiapradagalan-creator/act2.git
cd act2

# 2. Crear el entorno virtual, instalar dependencias y ejecutar TODO el pipeline
python run_pipeline.py --setup
```

(En Linux/macOS usa `python3`.) Con esto se ejecutan, en orden y parandoo ante el primer error:

1. Validación de datos (contrato + dataset Iris)
2. Entrenamiento baseline (`train.py`)
3. Experimentos comparativos (`experimentos.py`)
4. Validación del modelo (`validar_modelo.py`)
5. Reporte final (`reporte.py`)

Ejecuciones posteriores (el entorno ya existe):

```bash
python run_pipeline.py
```

### Alternativa con Makefile

```bash
make setup   # crea .venv e instala dependencias (una sola vez)
make all     # ejecuta todo el pipeline
make help    # lista todos los targets
make clean   # borra mlflow.db, mlruns/ y reportes generados
```

> En Windows, `make` requiere instalarse aparte (p. ej. `choco install make`).
> Si no lo tienes, usa `run_pipeline.py`, que no depende de nada adicional.

### Ejecución manual (paso a paso)

```bash
# Crear y activar el entorno virtual
python -m venv .venv
.venv\Scripts\activate          # Windows (PowerShell/CMD)
source .venv/bin/activate       # Linux/macOS

pip install -r requirements.txt

python validar_datos.py
python train.py
python experimentos.py
python validar_modelo.py
python reporte.py
```

## 4. Resultados esperados

Tras ejecutar el pipeline en la raíz del proyecto se generan automáticamente:

- `mlflow.db` y carpeta `mlruns/`: experimento `Proyecto_Final_Grado` con todas las ejecuciones y modelos.
- `tabla_comparativa_final.csv` y `tabla_comparativa_final.xlsx`: comparación de runs.
- `grafico_metricas_final.png`: gráfico de accuracy (train / test / CV).

Valores de referencia (semilla fija `random_state=42`): accuracy del baseline = 1.0000;
mejor accuracy en validación cruzada ≈ 0.95. Los pasos de validacion imprimen `[OK]` o `[FALLO]`
y terminan con código de salida distinto de cero si algo falla.

Para explorar los resultados en la interfaz web de MLflow:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
# abrir http://127.0.0.1:5000
```

## 5. Validación de datos

`validar_datos.py` usa únicamente la biblioteca estándar y lee las reglas del contrato.
Comprueba:

- **Tipos:** valores numéricos (se rechazan texto y booleanos).
- **Nulos:** rechaza `null/None`, `NaN`, infinito y campos obligatorios ausentes.
- **Rangos lógicos:** cada variable debe estar en su `[min, max]`.
- Campos no definidos en el contrato.

```bash
python validar_datos.py                          # validacion del contrato + 150 filas de Iris
python validar_datos.py --archivo entrada.json   # valida un registro o una lista de registros
```

Uso desde código (p. ej. en la futura API de inferencia):

```python
from validar_datos import validar_registro

errores = validar_registro({"sepal_length_cm": 5.1, "sepal_width_cm": 3.5,
                            "petal_length_cm": 1.4, "petal_width_cm": 0.2})
assert errores == []
```

## 6. Contrato de datos (`contrato_datos.json`)

Define los 4 campos de entrada, sus tipos, unidades y rangos, la salida del modelo
(`0 = setosa`, `1 = versicolor`, `2 = virginica`) y contiene:

- `ejemplo_entrada_valida`: registro aceptado.
- `ejemplo_entrada_invalida`: registro rechazado, con los motivos documentados.

| Campo | Tipo | Rango (cm) |
|---|---|---|
| `sepal_length_cm` | float | 4.0 – 8.5 |
| `sepal_width_cm` | float | 1.5 – 5.0 |
| `petal_length_cm` | float | 0.5 – 7.5 |
| `petal_width_cm` | float | 0.0 – 3.0 |

Cualquier cambio en el esquema debe reflejarse en este archivo y en su campo `version`.

## 7. Solucion de problemas

- **`python` no se reconoce (Windows):** reinstala Python marcando *Add Python to PATH* o usa `py run_pipeline.py --setup`.
- **PowerShell bloquea `activate`:** no es necesario activar el entorno si usas `run_pipeline.py`; o ejecuta `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.
- **Error con `pywin32` en Linux/macOS:** en `requirements.txt` debe figurar como `pywin32==312; sys_platform == "win32"` (ya corregido).
- **Falta `openpyxl` al generar el XLSX:** debe estar en `requirements.txt` (ya incluido).
- **Empezar de cero:** `make clean` (o borrar `mlflow.db` y `mlruns/`) y volver a ejecutar el pipeline.
