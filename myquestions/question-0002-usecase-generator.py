import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

def generar_caso_de_uso_clasificar_riesgo_binario():
    rng = np.random.default_rng()

    n_filas = int(rng.integers(12, 20))

    pasos = rng.normal(8000, 1500, n_filas)
    ejercicio = rng.normal(45, 12, n_filas)
    sueno = rng.normal(7, 1.2, n_filas)
    frecuencia = rng.normal(72, 8, n_filas)

    riesgo_cardiaco = (
        0.003 * pasos
        - 0.08 * ejercicio
        - 1.2 * sueno
        + 0.25 * frecuencia
        + rng.normal(0, 1.5, n_filas)
    )

    df = pd.DataFrame({
        "pasos_diarios": pasos,
        "minutos_ejercicio": ejercicio,
        "horas_sueno": sueno,
        "frecuencia_reposo": frecuencia,
        "riesgo_cardiaco": riesgo_cardiaco
    })

    umbral = float(np.median(riesgo_cardiaco))

    input_data = {
        "df": df.copy(),
        "target_col": "riesgo_cardiaco",
        "umbral": umbral
    }

    X = df.select_dtypes(include=[np.number]).drop(columns=["riesgo_cardiaco"])
    y = (df["riesgo_cardiaco"].to_numpy() >= umbral).astype(int)

    modelo = DecisionTreeClassifier(random_state=42)
    modelo.fit(X, y)
    output_data = modelo.predict(X)

    return input_data, output_data