
import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats

# =====================================================
# 1. CONFIGURACION GENERAL
# =====================================================

st.set_page_config(
    page_title="Monitoreo Agrícola",
    page_icon="🌱",
    layout="wide"
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
    '<p class="subtitle">Sistema de análisis del rendimiento '
    'de parcelas agrícolas mediante regresión lineal múltiple</p>',
    unsafe_allow_html=True
)

st.divider()


# =====================================================
# 4. CARGA Y LIMPIEZA DEL DATASET
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

    columnas_numericas = [
        "volumen_riego",
        "fertilizante_kg",
        columna_temperatura,
        "rendimiento_ton"
    ]

    if "tipo_tratamiento" not in df.columns:
        st.error("No se encontró la columna tipo_tratamiento.")
        st.stop()

    df = df[
        columnas_numericas + ["tipo_tratamiento"]
    ].copy()

    # Limpiar datos numéricos

    df[columnas_numericas] = df[columnas_numericas].apply(
        pd.to_numeric,
        errors="coerce"
    )

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    df = df.dropna(
        subset=columnas_numericas
    )

    df["tipo_tratamiento"] = (
        df["tipo_tratamiento"]
        .fillna("Sin dato")
        .astype(str)
    )

    if len(df) < 5:
        st.error("No hay suficientes registros para el análisis.")
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

    residuos = Y - predicciones


    # =================================================
    # 6. TARJETAS KPI
    # =================================================

    st.markdown(
        '<p class="section-title">📊 Indicadores principales</p>',
        unsafe_allow_html=True
    )

    rendimiento_promedio = df["rendimiento_ton"].mean()

    r2_ajustado = modelo.rsquared_adj

    p_valor = modelo.f_pvalue

    k1, k2, k3 = st.columns(3)

    k1.metric(
        "🌾 Rendimiento promedio global",
        f"{rendimiento_promedio:.2f} ton/ha"
    )

    k2.metric(
        "📈 R² ajustado del modelo",
        f"{r2_ajustado * 100:.2f}%"
    )

    k3.metric(
        "🧪 P-valor de la prueba inferencial",
        f"{p_valor:.3e}"
    )

    st.caption(
        "El p-valor corresponde a la prueba F de significancia global "
        "del modelo de regresión."
    )

    st.divider()


    # =================================================
    # 7. CONTROLES INTERACTIVOS
    # =================================================

    st.markdown(
        '<p class="section-title">🎛️ Filtros de datos</p>',
        unsafe_allow_html=True
    )

    filtro1, filtro2 = st.columns(2)

    with filtro1:

        tratamientos = sorted(
            df["tipo_tratamiento"].unique().tolist()
        )

        tratamientos_seleccionados = st.multiselect(
            "Seleccionar tipo de tratamiento",
            options=tratamientos,
            default=tratamientos
        )

    with filtro2:

        minimo = float(df["rendimiento_ton"].min())

        maximo = float(df["rendimiento_ton"].max())

        rango_rendimiento = st.slider(
            "Rango de rendimiento (ton/ha)",
            min_value=minimo,
            max_value=maximo,
            value=(minimo, maximo),
            step=0.1
        )

    # Aplicar filtros

    df_filtrado = df[
        (df["tipo_tratamiento"].isin(tratamientos_seleccionados)) &
        (df["rendimiento_ton"].between(
            rango_rendimiento[0],
            rango_rendimiento[1]
        ))
    ].copy()

    st.write(
        f"**Registros seleccionados:** {len(df_filtrado)} de {len(df)}"
    )

    if df_filtrado.empty:

        st.warning(
            "No existen registros con los filtros seleccionados."
        )

        st.stop()

    st.divider()


    # =================================================
    # 8. CUATRO GRAFICAS EN CUADRICULA 2 X 2
    # =================================================

    st.markdown(
        '<p class="section-title">📉 Análisis y diagnóstico visual</p>',
        unsafe_allow_html=True
    )

    st.write(
        "Las gráficas se actualizan automáticamente "
        "al modificar los filtros."
    )


    # -------------------------------------------------
    # GRAFICA 1: CORRELACION DE PEARSON
    # -------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        fig1, ax1 = plt.subplots(figsize=(7, 5))

        datos_correlacion = df_filtrado[
            columnas_numericas
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
            ax=ax1
        )

        ax1.set_title(
            "1. Matriz de correlación de Pearson",
            fontweight="bold"
        )

        plt.tight_layout()

        st.pyplot(
            fig1,
            use_container_width=True
        )

        plt.close(fig1)


    # -------------------------------------------------
    # GRAFICA 2: REGRESION PARCIAL
    # -------------------------------------------------

    with col2:

        fig2, ax2 = plt.subplots(figsize=(7, 5))

        if len(df_filtrado) >= 5:

            controles = sm.add_constant(
                df_filtrado[
                    [
                        "fertilizante_kg",
                        columna_temperatura
                    ]
                ]
            )

            modelo_riego = sm.OLS(
                df_filtrado["volumen_riego"],
                controles
            ).fit()

            modelo_rendimiento = sm.OLS(
                df_filtrado["rendimiento_ton"],
                controles
            ).fit()

            residuos_riego = modelo_riego.resid

            residuos_rendimiento = modelo_rendimiento.resid

            ax2.scatter(
                residuos_riego,
                residuos_rendimiento,
                alpha=0.7
            )

            if residuos_riego.nunique() > 1:

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

                ax2.plot(
                    x_linea,
                    pendiente * x_linea + intercepto,
                    linestyle="--"
                )

            ax2.set_xlabel("Riego ajustado")

            ax2.set_ylabel("Rendimiento ajustado")

            ax2.grid(alpha=0.25)

        else:

            ax2.text(
                0.5, 0.5,
                "Se necesitan al menos 5 registros.",
                ha="center",
                va="center"
            )

        ax2.set_title(
            "2. Regresión parcial: riego y rendimiento",
            fontweight="bold"
        )

        plt.tight_layout()

        st.pyplot(
            fig2,
            use_container_width=True
        )

        plt.close(fig2)


    # -------------------------------------------------
    # GRAFICA 3: COMPARACION DE TRATAMIENTOS
    # -------------------------------------------------

    col3, col4 = st.columns(2)

    with col3:

        fig3, ax3 = plt.subplots(figsize=(7, 5))

        sns.boxplot(
            data=df_filtrado,
            x="tipo_tratamiento",
            y="rendimiento_ton",
            ax=ax3
        )

        ax3.set_title(
            "3. Rendimiento por tipo de tratamiento",
            fontweight="bold"
        )

        ax3.set_xlabel("Tipo de tratamiento")

        ax3.set_ylabel("Rendimiento (ton/ha)")

        ax3.tick_params(
            axis="x",
            rotation=15
        )

        plt.tight_layout()

        st.pyplot(
            fig3,
            use_container_width=True
        )

        plt.close(fig3)


    # -------------------------------------------------
    # GRAFICA 4: DISTRIBUCION DE RESIDUOS
    # -------------------------------------------------

    with col4:

        fig4, ax4 = plt.subplots(figsize=(7, 5))

        X_filtrado = sm.add_constant(
            df_filtrado[variables]
        )

        predicciones_filtradas = modelo.predict(
            X_filtrado
        )

        residuos_filtrados = (
            df_filtrado["rendimiento_ton"] -
            predicciones_filtradas
        )

        sns.histplot(
            residuos_filtrados,
            kde=True,
            stat="density",
            ax=ax4,
            label="Residuos"
        )

        media = residuos_filtrados.mean()

        desviacion = residuos_filtrados.std(ddof=1)

        if desviacion > 0:

            x_normal = np.linspace(
                residuos_filtrados.min(),
                residuos_filtrados.max(),
                200
            )

            y_normal = stats.norm.pdf(
                x_normal,
                media,
                desviacion
            )

            ax4.plot(
                x_normal,
                y_normal,
                linestyle="--",
                label="Distribución normal"
            )

        ax4.set_title(
            "4. Distribución de los residuos",
            fontweight="bold"
        )

        ax4.set_xlabel("Residuos")

        ax4.set_ylabel("Densidad")

        ax4.legend()

        plt.tight_layout()

        st.pyplot(
            fig4,
            use_container_width=True
        )

        plt.close(fig4)


    # =================================================
    # 9. PIE DE PAGINA
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
        "Verifica que esté en la misma carpeta que app_fase3.py."
    )

except Exception as e:

    st.error(
        f"Se produjo un error: {e}"
    )
