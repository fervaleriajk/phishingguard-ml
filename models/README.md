# Modelos generados desde Google Colab

Coloca aquí los archivos creados por las secciones 10 y 11 de `entrenamiento_phishing_optimizado_precision_accuracy.ipynb`:

- `modelo_texto.pkl` (obligatorio)
- `modelo_url.pkl` (obligatorio)
- `modelo_fusion.pkl` (obligatorio)
- `metricas_operativas.json` (opcional)
- `resultados.csv` (opcional)

Estos artefactos NO están incluidos en este paquete porque se entrenan con tu Google Drive.

**Seguridad:** nunca cargues con `joblib` archivos `.pkl` de fuentes desconocidas. Los archivos pueden ejecutar código durante la deserialización.

**Compatibilidad:** usa la misma versión de `scikit-learn` y una versión compatible de Python que usaste en Colab.
