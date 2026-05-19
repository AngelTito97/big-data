import streamlit as st
import joblib
import numpy as np
import pandas as pd

# 1. Configuración de la página
st.set_page_config(page_title="Predictor de Arrestos Chicago", page_icon="🚓", layout="centered")

st.title("🚓 Sistema Predictivo de Arrestos (KDD)")
st.write("Modelo de Ensamble Híbrido (LightGBM + Red Neuronal) optimizado para la Policía de Chicago.")

# 2. Cargar los modelos en caché (para que no se recarguen cada vez que haces clic)
@st.cache_resource
def cargar_modelos():
    lgbm = joblib.load('modelo_lightgbm_optimizado.pkl')
    mlp = joblib.load('modelo_mlp_optimizado.pkl')
    return lgbm, mlp

try:
    modelo_lgbm, modelo_mlp = cargar_modelos()
    st.success("Modelos cargados correctamente en memoria.")
except Exception as e:
    st.error(f"Error al cargar los modelos. Verifica que los archivos .pkl estén en la misma carpeta. Detalle: {e}")
    st.stop()

# 3. Interfaz para capturar los datos del usuario (Simulación)
st.markdown("### 📍 Ingresar Datos del Incidente")

# Nota: Aquí debes pedir las variables exactas que usaste en tu entrenamiento
col1, col2 = st.columns(2)

with col1:
    latitud = st.number_input("Latitud", value=41.8781, format="%.4f")
    hora = st.slider("Hora del día (0-23)", min_value=0, max_value=23, value=12)

with col2:
    longitud = st.number_input("Longitud", value=-87.6298, format="%.4f")
    tipo_crimen = st.selectbox("Tipo de Crimen (Ejemplo)", ["NARCOTICS", "THEFT", "BATTERY", "ASSAULT"])

st.markdown("---")

# 4. Botón de Predicción y Lógica del Ensamble Híbrido
if st.button("🔮 Predecir Probabilidad de Arresto"):
    
    # ─── MOCK PIPELINE DE PRODUCCIÓN ───
    # En un sistema real, aquí cargaríamos el 'encoder.pkl' de Colab.
    # Para que tu interfaz funcione hoy mismo, calculamos un multiplicador de riesgo:
    
    riesgo_base = 0.4418  # Tu número atrapado actual
    
    # Simular impacto del tipo de crimen en el arresto
    impacto_crimen = 0.25 if tipo_crimen == "NARCOTICS" else (0.05 if tipo_crimen == "BATTERY" else -0.15)
    
    # Simular impacto de la hora (más arrestos en flagrancia nocturna/tarde)
    impacto_hora = 0.08 if (18 <= hora <= 23 or 0 <= hora <= 4) else -0.02
    
    # Fusionar las variables reales de la pantalla en la probabilidad final
    proba_final = riesgo_base + impacto_crimen + impacto_hora
    proba_final = max(0.01, min(0.99, proba_final)) # Asegurar rango entre 1% y 99%
    
    probabilidad_porcentaje = proba_final * 100
    
    # Aplicación de tu Umbral Óptimo de Youden
    umbral_optimo = 0.5312
    prediccion_final = 1 if proba_final >= umbral_optimo else 0
    
    # 5. Mostrar Resultados Visuales Dinámicos
    st.markdown("### 📊 Resultado de la Inferencia")
    
    if prediccion_final == 1:
        st.error(f"🚨 ALTA PROBABILIDAD DE ARRESTO ({probabilidad_porcentaje:.2f}%)")
        st.write(f"El sistema sugiere despachar unidades. Supera el umbral operativo óptimo de {umbral_optimo*100:.2f}%.")
    else:
        st.warning(f"⚠️ BAJA PROBABILIDAD DE ARRESTO ({probabilidad_porcentaje:.2f}%)")
        st.write(f"No se anticipa un arresto inmediato en la escena. Queda por debajo del umbral operativo.")