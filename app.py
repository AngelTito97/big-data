import streamlit as st
import joblib
import numpy as np
import pandas as pd

# 1. Configuración de la página (layout ancho para acomodar más campos)
st.set_page_config(page_title="Predictor de Arrestos Chicago", page_icon="🚓", layout="wide")

st.title("🚓 Sistema Predictivo de Arrestos (KDD)")
st.write("Modelo de Ensamble Híbrido (LightGBM + Red Neuronal) optimizado para la Policía de Chicago.")

# 2. Cargar los modelos y el preprocesador en caché
@st.cache_resource
def cargar_componentes():
    lgbm = joblib.load('modelo_lightgbm_optimizado (1).pkl')
    mlp = joblib.load('modelo_mlp_optimizado (1).pkl')
    transformador = joblib.load('preprocesador_chicago (1).pkl')
    return lgbm, mlp, transformador

try:
    modelo_lgbm, modelo_mlp, preprocessor = cargar_componentes()
    st.success("Componentes del modelo cargados correctamente en memoria.")
except Exception as e:
    st.error(f"Error al cargar los componentes. Verifica que los archivos .pkl estén en la misma carpeta. Detalle: {e}")
    st.stop()

# 3. Interfaz para capturar los datos del usuario (Diseño en 3 columnas)
# 3. Interfaz Minimalista (Solo pedimos 4 cosas)
st.markdown("### 📍 Ingresar Datos del Incidente")

col1, col2 = st.columns(2)

with col1:
    latitud = st.number_input("Latitud", value=41.8781, format="%.4f")
    longitud = st.number_input("Longitud", value=-87.6298, format="%.4f")

with col2:
    hora = st.slider("Hora del día", min_value=0, max_value=23, value=12)
    tipo_crimen = st.selectbox(
    "Tipo de Crimen (Primary Type)", 
    [
        "THEFT", 
        "BATTERY", 
        "NARCOTICS", 
        "ASSAULT", 
        "CRIMINAL DAMAGE", 
        "BURGLARY", 
        "ROBBERY", 
        "MOTOR VEHICLE THEFT", 
        "DECEPTIVE PRACTICE", 
        "WEAPONS VIOLATION",
        "HOMICIDE"
    ]
)
st.markdown("---")

# 4. Botón de Predicción y Lógica
if st.button("🔮 Predecir Probabilidad de Arresto"):
    
    try:
        # --- RELLENO AUTOMÁTICO (Back-end) ---
        # Estas variables NO se piden en pantalla, el sistema las asume por defecto
        # Usamos los valores más comunes (Moda/Mediana) de Chicago
        mes_fijo = 6  # Asumimos que es Junio
        dia_semana_fijo = 4  # Asumimos que es Viernes (0=Lunes, 4=Viernes)
        periodo_dia_fijo = "Afternoon"
        fbi_code_fijo = "06" # Código común de robo
        loc_desc_fijo = "STREET" # La mayoría ocurre en la calle
        distrito_fijo = 11
        comunidad_fija = 25
        ward_fijo = 28
        x_coordinate = 1100000.0
        y_coordinate = 1900000.0
        
        # --- CONSTRUIR EL DATAFRAME EXACTO ---
        datos_crudos = pd.DataFrame([{
            'Latitude': latitud,               # Viene de la pantalla
            'Longitude': longitud,             # Viene de la pantalla
            'Hour': hora,                      # Viene de la pantalla
            'Primary Type': tipo_crimen,       # Viene de la pantalla
            
            'FBI Code': fbi_code_fijo,         # Relleno automático
            'Day_Period': periodo_dia_fijo,    # Relleno automático
            'DayOfWeek': dia_semana_fijo,      # Relleno automático
            'X Coordinate': x_coordinate,      # Relleno automático
            'Month': mes_fijo,                 # Relleno automático
            'Y Coordinate': y_coordinate,      # Relleno automático
            'Location Description': loc_desc_fijo, # Relleno automático
            'Community Area': comunidad_fija,  # Relleno automático
            'District': distrito_fijo,         # Relleno automático
            'Ward': ward_fijo                  # Relleno automático
        }])
        
        # TRADUCCIÓN E INFERENCIA
        X_nuevo_proc = preprocessor.transform(datos_crudos)
        
        # (Aquí sigue tu código de predicción igual que siempre...)
        proba_lgb = modelo_lgbm.predict_proba(X_nuevo_proc)[:, 1]
        proba_mlp = modelo_mlp.predict_proba(X_nuevo_proc)[:, 1]
        
        proba_ensamble = (proba_lgb + proba_mlp) / 2.0
        probabilidad_porcentaje = proba_ensamble[0] * 100
        
        umbral_optimo = 0.5312
        prediccion_final = 1 if proba_ensamble[0] >= umbral_optimo else 0
        
        # Resultados...
        if prediccion_final == 1:
            st.error(f"🚨 ALTA PROBABILIDAD DE ARRESTO ({probabilidad_porcentaje:.2f}%)")
        else:
            st.warning(f"⚠️ BAJA PROBABILIDAD DE ARRESTO ({probabilidad_porcentaje:.2f}%)")
            
    except Exception as e:
        st.error(f"Error técnico: {e}")