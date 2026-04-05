import numpy as np
import pandas as pd

def generar_caso_de_uso_obtener_maximo_por_categoria():
    rng = np.random.default_rng()

    categorias_base = np.array(["audio", "hogar", "libros", "ropa", "tecnologia"])
    n_filas = int(rng.integers(12, 20))

    categorias = rng.choice(categorias_base, size=n_filas, replace=True)
    valores = rng.normal(100, 25, n_filas).round(2)
    unidades = rng.integers(1, 8, size=n_filas)

    df = pd.DataFrame({
        "categoria": categorias,
        "valor_venta": valores,
        "unidades": unidades
    })

    input_data = {
        "df": df.copy(),
        "columna_categoria": "categoria",
        "columna_valor": "valor_venta"
    }

    resultado = (
        df.groupby("categoria")["valor_venta"]
        .max()
        .sort_index()
        .to_numpy()
    )

    output_data = np.array(resultado)

    return input_data, output_data