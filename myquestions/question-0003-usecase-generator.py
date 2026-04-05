import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

def generar_caso_de_uso_detectar_anomalias_maquina():
    rng = np.random.default_rng()

    n_normales = int(rng.integers(12, 18))
    n_anomalas = int(rng.integers(2, 5))

    normales = np.column_stack([
        rng.normal(50, 4, n_normales),
        rng.normal(70, 5, n_normales),
        rng.normal(30, 3, n_normales),
        rng.normal(100, 8, n_normales)
    ])

    anomalas = np.column_stack([
        rng.normal(90, 3, n_anomalas),
        rng.normal(120, 4, n_anomalas),
        rng.normal(10, 2, n_anomalas),
        rng.normal(160, 6, n_anomalas)
    ])

    datos = np.vstack([normales, anomalas])
    rng.shuffle(datos)

    df = pd.DataFrame(
        datos,
        columns=["vibracion", "temperatura", "presion", "consumo_electrico"]
    )

    contamination = n_anomalas / (n_normales + n_anomalas)

    input_data = {
        "df": df.copy(),
        "contamination": contamination
    }

    X = df.select_dtypes(include=[np.number])

    modelo = IsolationForest(
        contamination=contamination,
        random_state=42
    )
    modelo.fit(X)
    output_data = modelo.predict(X)

    return input_data, output_data