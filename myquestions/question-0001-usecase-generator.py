import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression

def generar_caso_de_uso_seleccionar_mejores_variables():
    rng = np.random.default_rng()

    n_filas = int(rng.integers(12, 20))
    k = int(rng.integers(1, 4))

    x1 = rng.normal(10, 2, n_filas)
    x2 = rng.normal(20, 3, n_filas)
    x3 = rng.normal(30, 4, n_filas)
    x4 = rng.normal(40, 5, n_filas)

    y = 4 * x1 - 3 * x3 + rng.normal(0, 0.5, n_filas)

    for arr in [x1, x2, x3, x4]:
        mascara = rng.random(n_filas) < 0.15
        arr[mascara] = np.nan

    df = pd.DataFrame({
        "humedad_suelo": x1,
        "ph_suelo": x2,
        "temperatura": x3,
        "lluvia": x4,
        "rendimiento": y
    })

    input_data = {
        "df": df.copy(),
        "target_col": "rendimiento",
        "k": k
    }

    X = df.drop(columns=["rendimiento"])
    X = X.select_dtypes(include=[np.number])
    y_target = df["rendimiento"].to_numpy()

    imputer = SimpleImputer(strategy="mean")
    X_imputed = imputer.fit_transform(X)

    selector = SelectKBest(score_func=f_regression, k=k)
    selector.fit(X_imputed, y_target)

    output_data = np.array(X.columns[selector.get_support()], dtype=str)

    return input_data, output_data