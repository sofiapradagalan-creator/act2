"""Validación de datos de entrada SIN librerías externas (solo biblioteca estándar).

Verifica, a partir del contrato de datos (contrato_datos.json):
  1. Tipos de datos correctos (numéricos; se rechazan texto y booleanos).
  2. Ausencia de valores nulos (None, NaN, infinito) y de campos faltantes.
  3. Rangos lógicos para cada variable.

Uso:
  python validar_datos.py                    # autotest del contrato + validación del dataset Iris
  python validar_datos.py --archivo x.json   # valida un registro (o lista de registros) JSON
"""
import json
import math
import sys
from pathlib import Path

RUTA_CONTRATO = Path(__file__).with_name("contrato_datos.json")


def cargar_contrato(ruta=RUTA_CONTRATO):
    with open(ruta, encoding="utf-8") as f:
        return json.load(f)


def _es_numero(valor):
    # bool es subclase de int en Python, pero True/False no son mediciones válidas.
    return isinstance(valor, (int, float)) and not isinstance(valor, bool)


def validar_registro(registro, contrato=None):
    """Devuelve una lista de errores (vacía si el registro es válido)."""
    contrato = contrato or cargar_contrato()
    campos = contrato["entrada"]["campos"]
    nombres = [c["nombre"] for c in campos]

    if not isinstance(registro, dict):
        return ["El registro debe ser un objeto JSON (diccionario campo -> valor)."]

    errores = []

    if not contrato["entrada"].get("campos_extra_permitidos", False):
        for extra in registro:
            if extra not in nombres:
                errores.append(f"{extra}: campo no definido en el contrato")

    for campo in campos:
        nombre = campo["nombre"]

        if nombre not in registro:
            if campo.get("obligatorio", True):
                errores.append(f"{nombre}: campo obligatorio ausente")
            continue

        valor = registro[nombre]

        # 1) Nulos
        if valor is None:
            errores.append(f"{nombre}: valor nulo no permitido")
            continue

        # 2) Tipo
        if not _es_numero(valor):
            errores.append(
                f"{nombre}: tipo incorrecto ({type(valor).__name__}), se esperaba un número"
            )
            continue

        # 3) NaN / infinito (también cuentan como valores nulos/ inválidos)
        if math.isnan(valor) or math.isinf(valor):
            errores.append(f"{nombre}: valor NaN o infinito no permitido")
            continue

        # 4) Rango lógico
        if not (campo["min"] <= valor <= campo["max"]):
            errores.append(
                f"{nombre}: valor {valor} fuera del rango lógico [{campo['min']}, {campo['max']}]"
            )

    return errores


def validar_lote(registros, contrato=None):
    """Valida una lista de registros. Devuelve {indice: [errores]} solo para los inválidos."""
    contrato = contrato or cargar_contrato()
    resultado = {}
    for i, reg in enumerate(registros):
        errs = validar_registro(reg, contrato)
        if errs:
            resultado[i] = errs
    return resultado


def autotest_contrato(contrato):
    """El ejemplo válido debe pasar y el inválido debe ser rechazado."""
    ok = True
    errs_valido = validar_registro(contrato["ejemplo_entrada_valida"], contrato)
    if errs_valido:
        ok = False
        print("[FALLO] El ejemplo válido del contrato fue rechazado:", errs_valido)
    else:
        print("[OK] Ejemplo válido del contrato: aceptado")

    errs_invalido = validar_registro(contrato["ejemplo_entrada_invalida"]["entrada"], contrato)
    if not errs_invalido:
        ok = False
        print("[FALLO] El ejemplo inválido del contrato fue aceptado")
    else:
        print(f"[OK] Ejemplo inválido del contrato: rechazado con {len(errs_invalido)} errores")
        for e in errs_invalido:
            print("     -", e)
    return ok


def validar_dataset_iris(contrato):
    """Valida las 150 filas del dataset real usado en el entrenamiento."""
    try:
        from sklearn.datasets import load_iris  # solo se necesita para cargar los datos
    except ImportError:
        print("[AVISO] scikit-learn no está instalado; se omite la validación del dataset.")
        return True

    datos = load_iris()
    nombres = contrato["entrada"]["orden_para_el_modelo"]
    registros = [dict(zip(nombres, (float(x) for x in fila))) for fila in datos.data]
    invalidos = validar_lote(registros, contrato)

    etiquetas_validas = {int(k) for k in contrato["salida"]["valores_posibles"]}
    etiquetas_malas = [i for i, y in enumerate(datos.target) if int(y) not in etiquetas_validas]

    if invalidos or etiquetas_malas:
        print(f"[FALLO] Dataset Iris: {len(invalidos)} filas inválidas, "
              f"{len(etiquetas_malas)} etiquetas fuera de rango")
        for i, errs in list(invalidos.items())[:10]:
            print(f"     fila {i}: {errs}")
        return False

    print(f"[OK] Dataset Iris: {len(registros)} filas validadas (tipos, nulos y rangos)")
    return True


def main(argv):
    contrato = cargar_contrato()

    if len(argv) >= 3 and argv[1] == "--archivo":
        with open(argv[2], encoding="utf-8") as f:
            contenido = json.load(f)
        registros = contenido if isinstance(contenido, list) else [contenido]
        invalidos = validar_lote(registros, contrato)
        if invalidos:
            for i, errs in invalidos.items():
                print(f"Registro {i}: INVÁLIDO")
                for e in errs:
                    print("  -", e)
            return 1
        print(f"{len(registros)} registro(s) válido(s)")
        return 0

    ok = autotest_contrato(contrato)
    ok = validar_dataset_iris(contrato) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
