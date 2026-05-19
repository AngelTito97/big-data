# 🚓 Sistema Predictivo de Arrestos - Chicago Crimes (KDD)

Este proyecto implementa una aplicación web interactiva desarrollada en **Streamlit** que utiliza una arquitectura de **Ensamble Híbrido (LightGBM + Red Neuronal MLP)** para predecir la probabilidad de que un incidente delictivo en la ciudad de Chicago resulte en un arresto efectivo. 

El sistema sigue rigurosamente la metodología **KDD (Knowledge Discovery in Databases)** y aborda problemas complejos de la ciencia de datos como el desbalance crítico de clases mediante **Aprendizaje Sensible al Costo (Cost-Sensitive Learning)** y optimización geométrica de umbrales.

---

## 📊 Rendimiento del Modelo Final

Tras aplicar el **Índice J de Youden**, calibramos el umbral analítico a **0.5312**, logrando un equilibrio óptimo para el despliegue operativo en seguridad pública:

| Modelo Clasificador | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Ensamble Híbrido (Umbral Óptimo)** | **0.8513** | **0.7000** | **0.6967** | **0.6983** | **0.8864** |

*El modelo logra capturar cerca del 70% de los arrestos reales (Recall) manteniendo una certeza del 70% en sus alertas (Precision), reduciendo drásticamente los falsos negativos operativos.*

---

## 📝 Documento de Requerimientos del Sistema (SRS)

### 1. Descripción General
El objetivo principal es evaluar las características de un incidente delictivo en tiempo real y predecir la probabilidad de un arresto inmediato, optimizando la toma de decisiones y el despliegue de unidades de patrullaje en cuadrantes críticos.

### 2. Requerimientos Funcionales (RF)
* **RF-01 (Captura de Datos):** El sistema debe permitir al usuario ingresar mediante la interfaz gráfica la latitud, longitud, hora del incidente y la tipología delictiva.
* **RF-02 (Procesamiento Híbrido):** El sistema debe procesar las entradas combinando las probabilidades matemáticas de los modelos base (`LightGBM` y `MLP Neural Network`).
* **RF-03 (Calibración por Umbral):** El sistema debe clasificar de forma binaria el arresto aplicando el umbral geométrico optimizado de $0.5312$.
* **RF-04 (Alertas Operativas):** Si la probabilidad supera el $53.12\%$, la interfaz debe cambiar dinámicamente a una alerta roja de alta prioridad. De lo contrario, mostrará un estado amarillo de baja prioridad.

### 3. Requerimientos No Funcionales (RNF)
* **RNF-01 (Interfaz de Usuario):** El software debe ser desplegado mediante una interfaz web ligera e intuitiva basada estrictamente en la librería *Streamlit*.
* **RNF-02 (Latencia de Inferencia):** El tiempo total de procesamiento desde el clic del usuario hasta el despliegue del resultado no debe exceder los 2 segundos.
* **RNF-03 (Persistencia y Arquitectura):** Los modelos entrenados deben cargarse en memoria RAM como objetos serializados (`.pkl`) mediante `joblib`, evitando reentrenamientos en producción.
* **RNF-04 (Aislamiento de Entorno):** La aplicación debe ejecutarse dentro de un entorno virtual de Python (`venv`) para mitigar conflictos de versiones de librerías.

---

## 🛠️ Requisitos e Instalación

Sigue estos pasos estructurados para clonar y ejecutar el proyecto de forma local:

### 1. Clonar el repositorio
```bash
git clone [https://github.com/AngelTito97/big-data.git](https://github.com/AngelTito97/big-data.git)
cd big-data

2. Crear y activar el entorno virtual
En Windows (PowerShell):

PowerShell
py -m venv venv
.\venv\Scripts\Activate.ps1
### 3. Instalar las dependencias técnicas
Bash
pip install -r requirements.txt
🚀 Ejecución de la Aplicación
Con el entorno virtual activo, levanta el servidor local de la interfaz gráfica ejecutando:

Bash
streamlit run app.py
Esto abrirá automáticamente tu navegador web en la dirección http://localhost:8501 para interactuar con el simulador predictivo.

📂 Estructura del Repositorio
app.py: Archivo principal con la lógica de la interfaz y la capa lógica del ensamble (soft-voting).

requirements.txt: Archivo de gestión de dependencias y versiones fijas de Python.

README.md: Este archivo unificado (Presentación + Requerimientos + Guía de usuario).

.gitignore: Filtro de exclusión para evitar subir archivos basura, temporales o entornos locales (venv/).

componente_lightgbm.pkl: Serialización del modelo Gradient Boosting.

componente_red_neuronal.pkl: Serialización de la Red Neuronal Artificial (MLP).