# 🛡️ PhishingGuard — Aplicación Streamlit de detección de phishing

Interfaz web para clasificar **correos**, **URLs** y **CSV de correos** mediante los modelos de tu notebook de Google Colab.

- **1 = phishing**
- **0 = no phishing / legítimo**
- Diseño multimodal: texto (TF-IDF + regresión logística), URL (Random Forest) y fusión ponderada
- Mantiene la división 60 % entrenamiento / 20 % validación / 20 % prueba del notebook de correos
- No visita URLs ni abre adjuntos. Solo extrae características textuales y léxicas
- Resultados orientativos; no sustituyen verificaciones de seguridad

## 1. Exportar modelos desde Google Colab

En el notebook incluido `notebooks/entrenamiento_phishing_optimizado_precision_accuracy.ipynb`, ejecuta hasta las **secciones 10 y 11**. Estas guardan en Google Drive (carpeta `Mi unidad/phishing/models/`) y descargan `models_phishing.zip`.

Extrae del ZIP y añade los siguientes archivos dentro de `models/` del presente proyecto:

```text
models/
├── modelo_texto.pkl
├── modelo_url.pkl
├── modelo_fusion.pkl
├── metricas_operativas.json       (opcional)
└── resultados.csv                 (opcional)
```

`features.py` ya se incluye en este repositorio. Es una copia **exacta** del módulo generado por el notebook optimizado, imprescindible para abrir `modelo_fusion.pkl` y para construir las mismas características de las URLs. No reemplaces ese archivo por otra versión sin volver a entrenar/exportar los modelos.

**IMPORTANTE:** los `.pkl` todavía no se incluyen porque no tengo acceso a los archivos entrenados que están en tu Google Drive. La aplicación NO fabricará predicciones ni métricas de ejemplo si faltan modelos.

### Compatibilidad de dependencias

En una celda de Colab, después del entrenamiento, ejecuta:

```python
import sklearn, numpy, pandas, joblib, sys
print('Python:', sys.version)
print('scikit-learn:', sklearn.__version__)
print('numpy:', numpy.__version__)
print('pandas:', pandas.__version__)
print('joblib:', joblib.__version__)
```

En `requirements.txt` reemplaza `scikit-learn>=1.5,<2` por `scikit-learn==VERSION_DE_COLAB`. Si no coincide la versión, los artefactos de scikit-learn pueden presentar errores al cargarse o al predecir. En Streamlit Community Cloud, al crear la app, elige en **Advanced settings** una versión de Python compatible con los paquetes del entorno de Colab.

## 2. Probar localmente (opcional)

Con Python instalado:

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

No se puede evaluar sin los archivos de `models/`. La app mostrará instrucciones en lugar de predecir cuando falten.

## 3. Publicar en GitHub

1. En GitHub, crea un repositorio (por ejemplo `phishingguard-ml`).
2. Extrae el ZIP de este proyecto y sube **el contenido de la carpeta `phishingguard_streamlit/`** a la raíz del repositorio. Comprueba que `streamlit_app.py` quede en la raíz, no dentro de otra carpeta.
3. Copia los modelos exportados en `models/`. Importa también la versión de `requirements.txt` ajustada a Colab.
4. Confirma los cambios.

### Si algún `.pkl` es demasiado grande

GitHub permite subir por navegador archivos de hasta **25 MiB**; por Git normal admite hasta **100 MiB** por archivo. Puede que `modelo_url.pkl` supere ese tamaño; revisa sus tamaños antes de subir. Una forma de reducirlos en **Colab** es volver a guardarlos comprimidos (no cambia los valores de los modelos):

```python
import joblib, os
for nombre, modelo in [
    ('modelo_texto', modelo_texto),
    ('modelo_url', modelo_url),
    ('modelo_fusion', fusion),
]:
    ruta = f'/content/drive/MyDrive/phishing/models/{nombre}.pkl'
    joblib.dump(modelo, ruta, compress=3)
    print(os.path.basename(ruta), round(os.path.getsize(ruta)/1e6, 2), 'MB')
```

Los archivos `.pkl` comprimidos conservan el mismo nombre y pueden abrirse mediante `joblib.load`. Si el archivo todavía excede el límite, necesitarás otro método de almacenamiento/distribución o un modelo más ligero; Git LFS puede funcionar con repositorios, pero confirma que tu despliegue recupera los archivos reales (no los punteros LFS). Evita subir datasets originales, correos privados, credenciales, archivos `.env` o contraseñas al repositorio.

## 4. Desplegar en Streamlit Community Cloud

1. Visita https://share.streamlit.io/ e inicia sesión con GitHub.
2. Haz clic en **Create app** → **Yup, I have an app**.
3. Selecciona el repositorio y la rama (por ejemplo `main`).
4. **Main file path:** `streamlit_app.py`.
5. Revisa la versión de Python en **Advanced settings** y pulsa **Deploy**.
6. Obtendrás un enlace público de tu aplicación con terminación `.streamlit.app`.

Documentación oficial: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy

## 5. Estructura del repositorio

```text
phishingguard_streamlit/
├── streamlit_app.py                # Interfaz gráfica
├── inferencia.py                   # Lógica de predicción y pruebas
├── features.py                     # MISMA versión usada en el entrenamiento
├── requirements.txt               # Dependencias
├── .streamlit/config.toml          # Colores y tema
├── models/                        # Artefactos desde Colab
│   └── README.md
├── notebooks/entrenamiento_phishing_optimizado_precision_accuracy.ipynb
├── tests/test_inferencia.py        # Pruebas de inferencia con modelos falsos
├── ejemplo_correos.csv            # Archivo ilustrativo
└── README.md                       # Instrucciones
```

## 6. Consideraciones de evaluación

- La clasificación del correo usa el umbral `modelo_fusion.umbral` que elegiste usando validación; **no siempre es 0.5**.
- El análisis **aislado de URL** usa el modelo URL y un umbral de 0.5 (separado de la fusión de correos).
- Si `modelo_fusion.peso_texto` es igual a 1, el modelo validado decidió usar solo texto; no alteres ese peso para desplegar sin volver a validar.
- La tabla de métricas viene de tu entrenamiento, si subes `models/resultados.csv`.
- Un sitio HTTPS o un resultado `0` no garantiza seguridad. Las señales mostradas son **heurísticas**, no explicaciones del algoritmo.
- No pegues información corporativa confidencial en una app pública: en Streamlit Cloud el texto se procesa en el servidor.
- El código no hace solicitudes HTTP a las URLs recibidas ni ejecuta archivos adjuntos.
- Solo carga modelos `.pkl` creados por ti. `joblib.load` no debe utilizarse con modelos no confiables.
