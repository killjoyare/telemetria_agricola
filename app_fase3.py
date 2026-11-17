
import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from sklearn.metrics import mean_squared_error

# =====================================================
# 1. CONFIGURACION GENERAL
# =====================================================

st.set_page_config(
    page_title="Monitoreo Agrícola",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# 2. ESTILOS VISUALES
# =====================================================

st.markdown("""
<style>

.stApp {
    background-color: #f4f7f2;
}

.main-title {
    color: #1b5e35;
    font-size: 34px;
    font-weight: 800;
    margin-bottom: 0;
}

.subtitle {
    color: #607568;
    font-size: 16px;
    margin-top: 5px;
}

.section-title {
    color: #205c39;
    font-size: 23px;
    font-weight: 700;
}

div[data-testid="stMetric"] {
    background-color: white;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #dce8dc;
}

div.stButton > button,
div.stFormSubmitButton > button {
    background-color: #267346;
    color: white;
    border-radius: 9px;
    border: none;
    font-weight: 600;
    padding: 10px 20px;
}

div.stButton > button:hover,
div.stFormSubmitButton > button:hover {
    background-color: #174e2d;
    color: white;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# 3. ENCABEZADO
# =====================================================

st.markdown(
    '<p class="main-title">🌱 MONITOREO AGRÍCOLA</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">Sistema inteligente de análisis y '
    'predicción del rendimiento de parcelas agrícolas</p>',
    unsafe_allow_html=True
)

st.divider()


# =====================================================
# 4. CARGA DEL DATASET
# =====================================================

@st.cache_data
def cargar_datos():
    return pd.read_csv("telemetria_parcelas.csv")


try:

    df = cargar_datos()

    # Detectar columna de temperatura

    posibles_temperaturas = [
        "temperatura_c",
        "temperatura",
        "temperatura_max",
        "temperatura_maxima"
    ]

    columna_temperatura = next(
        (
            col for col in posibles_temperaturas
            if col in df.columns
        ),
        None
    )

    if columna_temperatura is None:
        st.error("No se encontró la columna de temperatura.")
        st.stop()

    # Variables necesarias

    columnas = [
        "volumen_riego",
        "fertilizante_kg",
        columna_temperatura,
        "rendimiento_ton"
    ]

    if "tipo_tratamiento" in df.columns:
        columnas.append("tipo_tratamiento")

    df = df[columnas].copy()

    # Limpiar datos numéricos

    columnas_numericas = [
        "volumen_riego",
        "fertilizante_kg",
        columna_temperatura,
        "rendimiento_ton"
    ]

    df[columnas_numericas] = df[columnas_numericas].apply(
        pd.to_numeric,
        errors="coerce"
    )

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna(
        subset=columnas_numericas
    )

    if len(df) < 4:
        st.error("No hay suficientes registros para entrenar el modelo.")
        st.stop()


    # =================================================
    # 5. ENTRENAMIENTO DEL MODELO OLS
    # =================================================

    variables = [
        "volumen_riego",
        "fertilizante_kg",
        columna_temperatura
    ]

    X = df[variables]

    Y = df["rendimiento_ton"]

    X_modelo = sm.add_constant(X)

    modelo = sm.OLS(Y, X_modelo).fit()

    predicciones = modelo.predict(X_modelo)

    residuos = modelo.resid

    rmse = np.sqrt(
        mean_squared_error(Y, predicciones)
    )


    # =================================================
    # 6. BARRA LATERAL
    # =================================================

    st.sidebar.title("🌿 Panel de control")

    st.sidebar.success(
        "Dataset cargado correctamente"
    )

    st.sidebar.write("### Información del sistema")

    st.sidebar.write(
        f"**Registros:** {len(df)}"
    )

    st.sidebar.write(
        "**Modelo:** Regresión lineal múltiple"
    )

    st.sidebar.write(
        f"**Temperatura:** {columna_temperatura}"
    )

    st.sidebar.divider()

    st.sidebar.write("### Rangos de las variables")

    for variable in variables:

        minimo = df[variable].min()
        maximo = df[variable].max()

        st.sidebar.write(f"**{variable}**")

        st.sidebar.caption(
            f"{minimo:.2f} - {maximo:.2f}"
        )


    # =================================================
    # 7. INDICADORES DEL MODELO
    # =================================================

    st.markdown(
        '<p class="section-title">📊 Indicadores del modelo</p>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Parcelas analizadas",
        len(df)
    )

    c2.metric(
        "R²",
        f"{modelo.rsquared:.4f}"
    )

    c3.metric(
        "R² ajustado",
        f"{modelo.rsquared_adj:.4f}"
    )

    c4.metric(
        "RMSE",
        f"{rmse:.4f}"
    )

    st.caption(
        "Indicadores calculados utilizando los registros del dataset."
    )

    st.divider()


    # =================================================
    # 8. PANEL DE DIAGNOSTICO: CUATRO GRAFICAS
    # =================================================

    st.markdown(
        '<p class="section-title">📉 Panel de diagnóstico visual</p>',
        unsafe_allow_html=True
    )

    st.write(
        "Análisis estadístico de las variables y del comportamiento "
        "del modelo de regresión lineal múltiple."
    )

    fig, axes = plt.subplots(
        2, 2,
        figsize=(15, 10)
    )


    # -------------------------------------------------
    # GRAFICA 1: MATRIZ DE CORRELACION DE PEARSON
    # -------------------------------------------------

    datos_correlacion = df[
        [
            "volumen_riego",
            "fertilizante_kg",
            columna_temperatura,
            "rendimiento_ton"
        ]
    ]

    correlacion = datos_correlacion.corr(
        method="pearson"
    )

    sns.heatmap(
        correlacion,
        annot=True,
        cmap="YlGn",
        fmt=".2f",
        linewidths=0.5,
        ax=axes[0, 0]
    )

    axes[0, 0].set_title(
        "1. Matriz de correlación de Pearson",
        fontweight="bold"
    )


    # -------------------------------------------------
    # GRAFICA 2: REGRESION PARCIAL
    # -------------------------------------------------

    controles = sm.add_constant(
        df[
            [
                "fertilizante_kg",
                columna_temperatura
            ]
        ]
    )

    modelo_riego = sm.OLS(
        df["volumen_riego"],
        controles
    ).fit()

    modelo_rendimiento = sm.OLS(
        df["rendimiento_ton"],
        controles
    ).fit()

    residuos_riego = modelo_riego.resid

    residuos_rendimiento = modelo_rendimiento.resid

    axes[0, 1].scatter(
        residuos_riego,
        residuos_rendimiento,
        alpha=0.7
    )

    pendiente, intercepto = np.polyfit(
        residuos_riego,
        residuos_rendimiento,
        1
    )

    x_linea = np.linspace(
        residuos_riego.min(),
        residuos_riego.max(),
        100
    )

    axes[0, 1].plot(
        x_linea,
        pendiente * x_linea + intercepto,
        linestyle="--"
    )

    axes[0, 1].set_title(
        "2. Regresión parcial: riego y rendimiento",
        fontweight="bold"
    )

    axes[0, 1].set_xlabel(
        "Riego ajustado"
    )

    axes[0, 1].set_ylabel(
        "Rendimiento ajustado"
    )

    axes[0, 1].grid(alpha=0.25)


    # -------------------------------------------------
    # GRAFICA 3: COMPARACION DE TRATAMIENTOS
    # -------------------------------------------------

    if "tipo_tratamiento" in df.columns:

        sns.boxplot(
            data=df,
            x="tipo_tratamiento",
            y="rendimiento_ton",
            ax=axes[1, 0]
        )

        axes[1, 0].set_title(
            "3. Rendimiento por tipo de tratamiento",
            fontweight="bold"
        )

        axes[1, 0].set_xlabel(
            "Tipo de tratamiento"
        )

        axes[1, 0].set_ylabel(
            "Rendimiento (ton/ha)"
        )

        axes[1, 0].tick_params(
            axis="x",
            rotation=15
        )

    else:

        axes[1, 0].text(
            0.5,
            0.5,
            "No existe la columna tipo_tratamiento",
            ha="center",
            va="center"
        )

        axes[1, 0].set_title(
            "3. Comparación de tratamientos"
        )


    # -------------------------------------------------
    # GRAFICA 4: HISTOGRAMA DE RESIDUOS
    # -------------------------------------------------

    sns.histplot(
        residuos,
        kde=True,
        stat="density",
        ax=axes[1, 1],
        label="Residuos"
    )

    media = residuos.mean()

    desviacion = residuos.std(ddof=1)

    if desviacion > 0:

        x_normal = np.linspace(
            residuos.min(),
            residuos.max(),
            200
        )

        y_normal = stats.norm.pdf(
            x_normal,
            media,
            desviacion
        )

        axes[1, 1].plot(
            x_normal,
            y_normal,
            linestyle="--",
            label="Distribución normal"
        )

    axes[1, 1].set_title(
        "4. Distribución de los residuos",
        fontweight="bold"
    )

    axes[1, 1].set_xlabel(
        "Residuos"
    )

    axes[1, 1].set_ylabel(
        "Densidad"
    )

    axes[1, 1].legend()


    # -------------------------------------------------
    # PRESENTACION DEL PANEL
    # -------------------------------------------------

    fig.suptitle(
        "ANÁLISIS Y DIAGNÓSTICO DEL MODELO OLS",
        fontsize=16,
        fontweight="bold"
    )

    plt.tight_layout(
        rect=[0, 0, 1, 0.95],
        h_pad=3.0,
        w_pad=2.0
    )

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)

    st.caption(
        "Panel de diagnóstico: correlación, regresión parcial, "
        "comparación de tratamientos y distribución de residuos."
    )

    st.divider()


    # =================================================
    # 9. FORMULARIO DE PREDICCION
    # =================================================

    st.markdown(
        '<p class="section-title">🔮 Predicción del rendimiento</p>',
        unsafe_allow_html=True
    )

    st.write(
        "Introduce las condiciones de una parcela para estimar "
        "su rendimiento agrícola."
    )

    with st.form("formulario_prediccion"):

        col1, col2, col3 = st.columns(3)

        with col1:

            riego = st.number_input(
                "💧 Volumen de riego (m³/ha)",
                min_value=0.0,
                value=150.0,
                step=5.0
            )

        with col2:

            fertilizante = st.number_input(
                "🌾 Fertilizante (kg)",
                min_value=0.0,
                value=80.0,
                step=5.0
            )

        with col3:

            temperatura = st.number_input(
                "🌡️ Temperatura (°C)",
                min_value=0.0,
                value=28.0,
                step=1.0
            )

        ejecutar = st.form_submit_button(
            "🔍 Calcular rendimiento",
            use_container_width=True
        )


    # =================================================
    # 10. HISTORIAL DE PREDICCIONES
    # =================================================

    if "historial" not in st.session_state:

        st.session_state.historial = []


    # =================================================
    # 11. RESULTADOS DE LA PREDICCION
    # =================================================

    if ejecutar:

        nuevos_datos = pd.DataFrame({
            "volumen_riego": [riego],
            "fertilizante_kg": [fertilizante],
            columna_temperatura: [temperatura]
        })

        nuevos_datos = sm.add_constant(
            nuevos_datos,
            has_constant="add"
        )

        resultado = modelo.get_prediction(
            nuevos_datos
        )

        resumen = resultado.summary_frame(
            alpha=0.05
        )

        rendimiento = resumen["mean"].iloc[0]

        limite_inferior = resumen["obs_ci_lower"].iloc[0]

        limite_superior = resumen["obs_ci_upper"].iloc[0]


        # Guardar historial

        st.session_state.historial.append({

            "Volumen de riego": riego,

            "Fertilizante": fertilizante,

            "Temperatura": temperatura,

            "Rendimiento estimado": round(
                rendimiento, 2
            ),

            "Límite inferior": round(
                limite_inferior, 2
            ),

            "Límite superior": round(
                limite_superior, 2
            )

        })


        st.divider()

        st.markdown(
            '<p class="section-title">📈 Resultado de la predicción</p>',
            unsafe_allow_html=True
        )

        r1, r2, r3 = st.columns(3)

        r1.metric(
            "Rendimiento estimado",
            f"{rendimiento:.2f} ton/ha"
        )

        r2.metric(
            "Límite inferior (95%)",
            f"{limite_inferior:.2f} ton/ha"
        )

        r3.metric(
            "Límite superior (95%)",
            f"{limite_superior:.2f} ton/ha"
        )

        st.success(
            "La predicción del rendimiento se realizó correctamente."
        )

        st.info(
            "El intervalo representa la predicción del 95% "
            "para una observación individual."
        )


        # Comprobar rangos

        fuera_de_rango = []

        valores = {

            "volumen_riego": riego,

            "fertilizante_kg": fertilizante,

            columna_temperatura: temperatura

        }

        for nombre, valor in valores.items():

            minimo = df[nombre].min()

            maximo = df[nombre].max()

            if valor < minimo or valor > maximo:

                fuera_de_rango.append(
                    f"{nombre}: {minimo:.2f} - {maximo:.2f}"
                )

        if fuera_de_rango:

            st.warning(
                "Algunos valores están fuera del rango observado "
                "en el dataset. La predicción puede ser menos confiable.\n\n"
                + "\n".join(fuera_de_rango)
            )


    # =================================================
    # 12. HISTORIAL Y DESCARGA
    # =================================================

    st.divider()

    st.markdown(
        '<p class="section-title">📋 Historial de predicciones</p>',
        unsafe_allow_html=True
    )

    if st.session_state.historial:

        historial_df = pd.DataFrame(
            st.session_state.historial
        )

        st.dataframe(
            historial_df,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "📥 Descargar historial CSV",

            data=historial_df.to_csv(
                index=False
            ).encode("utf-8-sig"),

            file_name="historial_predicciones.csv",

            mime="text/csv"
        )

        if st.button("🗑️ Limpiar historial"):

            st.session_state.historial = []

            st.rerun()

    else:

        st.info(
            "Todavía no hay predicciones registradas. "
            "Realiza una predicción para comenzar el historial."
        )


    # =================================================
    # 13. ECUACION Y RESUMEN ESTADISTICO
    # =================================================

    st.divider()

    st.markdown(
        '<p class="section-title">🧮 Información estadística</p>',
        unsafe_allow_html=True
    )

    coeficientes = modelo.params

    ecuacion = (

        f"Rendimiento = {coeficientes['const']:.4f}"

        f" + ({coeficientes['volumen_riego']:.4f} × Riego)"

        f" + ({coeficientes['fertilizante_kg']:.4f} × Fertilizante)"

        f" + ({coeficientes[columna_temperatura]:.4f} × Temperatura)"

    )

    st.code(ecuacion)

    with st.expander("Consultar resumen estadístico OLS"):

        st.text(
            modelo.summary()
        )

    with st.expander("Consultar dataset completo"):

        st.dataframe(
            df,
            use_container_width=True
        )


    # =================================================
    # 14. PIE DE PAGINA
    # =================================================

    st.divider()

    st.markdown(
        """
        <center>
        🌱 Sistema de Monitoreo Agrícola
        <br>
        Regresión Lineal Múltiple
        </center>
        """,
        unsafe_allow_html=True
    )


except FileNotFoundError:

    st.error(
        "No se encontró telemetria_parcelas.csv. "
        "Verifica que esté en la misma carpeta."
    )

except Exception as e:

    st.error(
        f"Se produjo un error: {e}"
    )
