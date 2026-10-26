"""PhishingGuard — app de Streamlit para modelos exportados desde Google Colab.

No visita URLs, no descarga adjuntos y no almacena correos en un backend.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from inferencia import analizar_correo, analizar_url, obtener_urls


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
NECESARIOS = ("modelo_texto.pkl", "modelo_url.pkl", "modelo_fusion.pkl")

st.set_page_config(page_title="PhishingGuard | Detector de phishing",
                   page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.block-container {max-width: 1190px; padding-top: 1.6rem;}
.hero {padding: 1.7rem 1.9rem; border: 1px solid #26455f; border-radius: 18px;
       background: linear-gradient(125deg, #0e243c, #142f40 70%, #103e49); margin-bottom: 1.1rem;}
.hero h1 {color:#f6fbff; margin:0; font-size:2.2rem; letter-spacing:-.025em;}
.hero p {color:#c1d9e7; margin:.45rem 0 0; font-size:1.03rem;}
.eyebrow {text-transform:uppercase; letter-spacing:.17em; color:#58d8cb; font-size:.77rem; font-weight:700;}
.score {font-size:2.4rem; font-weight:800; letter-spacing:-.04em;}
.safe {color:#4fe0ac;} .danger {color:#ff808f;}
.small-note {font-size:.85rem; opacity:.8;}
[data-testid="stMetricValue"] {font-size:1.35rem;}
</style>""", unsafe_allow_html=True)

st.markdown("""<div class="hero"><span class="eyebrow">Análisis de ciberseguridad · Machine Learning</span>
<h1>🛡️ PhishingGuard</h1><p>Detección de phishing en correos y enlaces mediante modelos de texto, URLs y fusión multimodal.</p>
</div>""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Cargando los modelos entrenados...")
def cargar_modelos():
    # Solo se cargan archivos entregados por el propietario de este repositorio.
    return tuple(joblib.load(MODEL_DIR / archivo) for archivo in NECESARIOS)


@st.cache_data

def leer_informe():
    ruta = MODEL_DIR / "resultados.csv"
    if not ruta.is_file():
        return None
    return pd.read_csv(ruta)


def mostrar_resultado(res, contexto: str):
    etiqueta = "PHISHING DETECTADO" if res.clase == 1 else "NO PHISHING DETECTADO"
    color = "danger" if res.clase else "safe"
    st.markdown(f'<div class="score {color}">{"⚠️" if res.clase else "✅"} {etiqueta}</div>',
                unsafe_allow_html=True)
    st.progress(min(1.0, max(0.0, res.score)), text=f"Puntuación de riesgo estimada: {res.score:.1%}")
    a, b, c = st.columns(3)
    a.metric("Clasificación", str(res.clase), help="1 = phishing; 0 = no phishing")
    b.metric("Umbral de decisión", f"{res.umbral:.1%}")
    c.metric("URLs detectadas", len(res.urls))
    if contexto == "correo":
        i, j = st.columns(2)
        i.metric("Modelo de texto", f"{res.score_texto:.1%}")
        j.metric("Modelo de URL", "Sin enlaces" if res.score_url is None else f"{res.score_url:.1%}")
    st.caption("Resultado orientativo de un modelo estadístico; no confirma que un enlace o mensaje sea seguro.")
    if res.senales_texto:
        st.markdown("**Señales textuales observadas**")
        for nombre, palabras in res.senales_texto.items():
            st.write(f"• {nombre}: {', '.join(palabras[:5])}")
    if res.urls:
        st.markdown("**Análisis de enlaces**")
        for n, url in enumerate(res.urls, 1):
            with st.expander(f"Enlace {n}: {url[:100]}", expanded=n == 1 and len(res.urls) == 1):
                senales = res.senales_urls.get(url, [])
                if senales:
                    for razon in senales:
                        st.write(f"• {razon}")
                else:
                    st.write("No se detectaron las señales heurísticas incluidas en este prototipo.")
                st.caption("Nunca abrimos ni visitamos automáticamente esta URL.")


with st.sidebar:
    st.subheader("⚙️ Modelo desplegado")
    st.write("**Convención:** `1` phishing · `0` legítimo")
    faltan = [nombre for nombre in NECESARIOS if not (MODEL_DIR / nombre).is_file()]
    if faltan:
        st.error("Faltan modelos entrenados")
    else:
        st.success("Modelos disponibles")
    st.divider()
    st.caption("Entrenamiento: SpaPhish (correos) y LegitPhish (URLs). División de correos: 60/20/20.")
    st.caption("Este prototipo no abre sitios externos, no lee tu bandeja de entrada ni envía información a otros servicios por código propio. Si se publica en la nube, el texto sí se procesa en el servidor del despliegue.")

if faltan:
    st.warning("La interfaz está preparada, pero necesita tus modelos exportados desde Colab para hacer predicciones reales.")
    st.markdown("Sube a la carpeta `models/` de tu repositorio: `modelo_texto.pkl`, `modelo_url.pkl` y `modelo_fusion.pkl`.")
    st.markdown("Opcionalmente añade `metricas_operativas.json` y `resultados.csv` para visualizar detalles del entrenamiento.")
    with st.expander("¿Dónde encuentro estos archivos?"):
        st.write("Al ejecutar las secciones 10 y 11 del notebook optimizado, se genera `models_phishing.zip` y se guardan los modelos en Google Drive, dentro de `phishing/models/`.")
    st.stop()

try:
    modelo_texto, modelo_url, modelo_fusion = cargar_modelos()
except Exception as exc:
    st.error("No fue posible cargar los modelos. Revisa que uses la misma versión de scikit-learn y Python que en Colab.")
    st.code(f"{type(exc).__name__}: {exc}")
    st.stop()

meta_path = MODEL_DIR / "metricas_operativas.json"
try:
    metadata = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
except (ValueError, OSError):
    metadata = {}

with st.sidebar:
    peso = float(getattr(modelo_fusion, "peso_texto", 1.0))
    umbral = float(getattr(modelo_fusion, "umbral", 0.5))
    st.metric("Umbral del modelo multimodal", f"{umbral:.1%}")
    st.write(f"Peso texto: **{peso:.0%}** · peso URLs: **{1-peso:.0%}**")
    if peso >= 0.999:
        st.info("La validación seleccionó solo texto en la fusión. El análisis individual de URLs sigue disponible.")

correo_tab, url_tab, lote_tab, info_tab = st.tabs(["✉️ Analizar correo", "🔗 Analizar URL", "📁 Análisis por lotes", "📊 Sobre el modelo"])

with correo_tab:
    st.subheader("¿Este correo puede ser phishing?")
    st.write("Introduce el **asunto** y el **cuerpo** del correo. Puedes agregar enlaces que no aparezcan en el texto.")
    with st.form("form_correo"):
        asunto = st.text_input("Asunto", placeholder="Ej.: Confirmación de cuenta")
        cuerpo = st.text_area("Contenido del mensaje", height=170,
                              placeholder="Pega aquí un correo de prueba, preferiblemente sin datos personales.")
        extra = st.text_area("URLs adicionales (opcional, una por línea)", height=75)
        enviar = st.form_submit_button("🛡️ Analizar correo", type="primary", use_container_width=True)
    if enviar:
        try:
            with st.spinner("Evaluando el mensaje..."):
                resultado = analizar_correo(modelo_texto, modelo_url, modelo_fusion, asunto, cuerpo, extra)
            mostrar_resultado(resultado, "correo")
        except Exception as exc:
            st.error(f"No se pudo analizar el correo: {exc}")

with url_tab:
    st.subheader("Evaluación individual de una URL")
    st.write("Analiza características estructurales del enlace sin visitarlo. Este modo utiliza exclusivamente el modelo entrenado con LegitPhish.")
    with st.form("form_url"):
        url = st.text_input("URL", placeholder="https://www.example.org/consulta")
        enviar_url = st.form_submit_button("🔎 Analizar URL", type="primary", use_container_width=True)
    if enviar_url:
        try:
            resultado_url = analizar_url(modelo_url, url)
            mostrar_resultado(resultado_url, "url")
        except Exception as exc:
            st.error(f"No se pudo analizar la URL: {exc}")
    st.caption("Umbral de URL: 0.50. Es independiente del umbral multimodal de los correos.")

with lote_tab:
    st.subheader("Procesar un CSV de correos")
    st.write("Formato esperado: columnas `subject` y `body` (opcional `urls`). También se acepta una columna `texto`.")
    st.caption("Máximo 300 correos y 3 MB. La descarga incluye solo puntuaciones, etiquetas e índices de fila, no el contenido original.")
    plantilla = "subject,body,urls\nAviso de reunión,Te comparto el resumen de la reunión,\nActualización urgente,Confirma tus datos ingresando al enlace,https://example.org/confirmar\n"
    st.download_button("📄 Descargar CSV de ejemplo", plantilla.encode("utf-8"), "ejemplo_correos.csv", "text/csv")
    archivo = st.file_uploader("Adjunta tu archivo CSV", type=["csv"], key="lote_csv")
    if archivo is not None:
        try:
            if archivo.size > 3_000_000:
                raise ValueError("El archivo supera 3 MB.")
            datos = pd.read_csv(archivo, encoding="utf-8-sig", sep=None, engine="python")
            if len(datos) > 300:
                raise ValueError("El archivo tiene más de 300 filas.")
            nombres = {str(c).strip().lower(): c for c in datos.columns}
            if "texto" not in nombres and not {"subject", "body"}.issubset(nombres):
                raise ValueError("Debes incluir 'subject' y 'body', o 'texto'.")
            st.info(f"Archivo válido: {len(datos)} filas. El análisis se inicia al pulsar el botón.")
            if st.button("▶️ Analizar CSV", type="primary"):
                registros = []
                progreso = st.progress(0)
                for k, (_, fila) in enumerate(datos.iterrows()):
                    if "texto" in nombres:
                        asunto, cuerpo = "", str(fila[nombres["texto"]]) if pd.notna(fila[nombres["texto"]]) else ""
                    else:
                        a = fila[nombres["subject"]]
                        b = fila[nombres["body"]]
                        asunto = str(a) if pd.notna(a) else ""
                        cuerpo = str(b) if pd.notna(b) else ""
                    campo_urls = fila[nombres["urls"]] if "urls" in nombres else ""
                    extra = str(campo_urls) if pd.notna(campo_urls) else ""
                    if not (asunto.strip() or cuerpo.strip()):
                        registros.append({"fila": k+1, "clase": "", "resultado": "sin texto", "riesgo": "", "num_urls": 0})
                    else:
                        r = analizar_correo(modelo_texto, modelo_url, modelo_fusion, asunto, cuerpo, extra)
                        registros.append({"fila": k+1, "clase": r.clase, "resultado": "phishing" if r.clase else "no phishing",
                                          "riesgo": round(r.score, 5), "num_urls": len(r.urls)})
                    progreso.progress((k+1) / max(len(datos), 1))
                salida = pd.DataFrame(registros)
                st.dataframe(salida, hide_index=True, use_container_width=True)
                st.download_button("⬇️ Descargar resultados CSV", salida.to_csv(index=False).encode("utf-8-sig"),
                                   "predicciones_phishingguard.csv", "text/csv", type="primary")
        except Exception as exc:
            st.error(f"No se pudo procesar el archivo CSV: {exc}")

with info_tab:
    st.subheader("Acerca de PhishingGuard")
    st.write("**Sistema de detección de riesgo de phishing** con un clasificador de texto (TF-IDF + regresión logística), un clasificador de URLs (Random Forest) y fusión ponderada seleccionada en validación.")
    st.write("**Etiquetas:** `1` phishing · `0` no phishing.")
    st.write("**División del dataset de correos:** 60 % entrenamiento · 20 % validación · 20 % prueba.")
    st.write("**Nota metodológica:** Las puntuaciones no están necesariamente calibradas como probabilidades reales; las explicaciones son indicadores heurísticos, no interpretaciones causales del modelo.")
    tabla = leer_informe()
    if tabla is not None:
        st.markdown("#### Evaluación exportada desde Colab")
        st.dataframe(tabla, hide_index=True, use_container_width=True)
        st.caption("Las métricas reflejan la evaluación reportada por el notebook, no el rendimiento verificado en producción.")
    else:
        st.info("Agrega `models/resultados.csv` si quieres mostrar las métricas reales del entrenamiento.")
    if metadata:
        st.markdown("#### Configuración de inferencia")
        claves = ("texto_seleccionado", "peso_texto", "peso_url", "umbral_decision", "n_train", "n_val", "n_test")
        st.json({k: metadata[k] for k in claves if k in metadata})
    st.warning("Uso académico y demostrativo. Un resultado 'no phishing' no garantiza seguridad. Evita subir correos corporativos o datos personales a una app pública.")
