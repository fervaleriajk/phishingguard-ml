"""Funciones compartidas entre Colab (entrenamiento) y Streamlit (app)."""
import re
import math
import unicodedata
from collections import Counter
from urllib.parse import urlparse
import pandas as pd

URL_REGEX = re.compile(r"(?:https?://|www\.)[^\s<>\"'\)\]]+", re.IGNORECASE)

ACORTADORES = {"bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd",
               "buff.ly", "cutt.ly", "rb.gy", "shorturl.at"}
TLD_SOSPECHOSOS = {"xyz", "top", "club", "click", "link", "info", "online", "site",
                   "tk", "ml", "ga", "cf", "gq", "work", "support", "live", "icu", "buzz"}
PALABRAS_URL = ["login", "signin", "verify", "verifica", "secure", "seguro", "account",
                "cuenta", "update", "actualiza", "bank", "banco", "password", "clave",
                "confirm", "pago", "pay", "wallet", "suspend", "bloque", "recover", "acceso"]

EXT_RIESGOSAS = {".exe", ".scr", ".js", ".vbs", ".bat", ".cmd", ".lnk", ".iso", ".jar",
                 ".zip", ".rar", ".7z", ".html", ".htm", ".docm", ".xlsm", ".pptm", ".msi"}

INDICADORES_TEXTO = {
    "Lenguaje de urgencia": ["urgente", "inmediat", "ultimo aviso", "24 horas", "48 horas",
                             "hoy mismo", "de inmediato", "expira", "vence", "cuanto antes",
                             "ultima oportunidad", "lo antes posible"],
    "Solicitud de credenciales / datos sensibles": ["contrasena", "clave", "usuario", "password",
                             "verifica tu cuenta", "confirma tus datos", "datos bancarios",
                             "tarjeta", "cvv", "numero de cuenta", "iniciar sesion",
                             "actualiza tus datos", "valida tu"],
    "Amenaza o consecuencia negativa": ["suspendida", "bloqueada", "cancelada", "cerrada",
                             "sancion", "multa", "penalidad", "acceso restringido",
                             "desactivada", "suspension"],
    "Lenguaje de autoridad": ["banco", "soporte tecnico", "departamento de", "administrador",
                             "sunat", "gerencia", "recursos humanos", "seguridad informatica",
                             "equipo de seguridad"],
    "Gancho de premio / beneficio": ["ganado", "premio", "gratis", "sorteo", "reembolso",
                             "herencia", "felicidades", "beneficio"],
    "Llamado a hacer clic / abrir adjunto": ["haz clic", "haga clic", "clic aqui", "click",
                             "pulsa", "descarga el adjunto", "abre el archivo", "ingresa aqui"],
}


def _norm(texto):
    texto = unicodedata.normalize("NFD", str(texto).lower())
    return "".join(c for c in texto if unicodedata.category(c) != "Mn")


def extract_urls(texto):
    urls = [u.rstrip(".,;:") for u in URL_REGEX.findall(str(texto or ""))]
    return list(dict.fromkeys(urls))


def _entropia(s):
    if not s:
        return 0.0
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in Counter(s).values())


def url_features(url):
    u = str(url).strip()
    con_esquema = bool(re.match(r"^[a-z]+://", u, re.I))
    p = urlparse(u if con_esquema else "http://" + u)
    host = (p.hostname or "").lower()
    path = p.path or ""
    partes = [x for x in host.split(".") if x]
    es_ip = bool(re.fullmatch(r"\d{1,3}(\.\d{1,3}){3}", host))
    tld = partes[-1] if partes else ""
    low = u.lower()
    try:
        tiene_puerto = int(p.port is not None)
    except ValueError:
        tiene_puerto = 1
    n = max(len(u), 1)
    n_dig = sum(c.isdigit() for c in u)
    return {
        "longitud": len(u),
        "longitud_dominio": len(host),
        "longitud_path": len(path),
        "n_puntos": u.count("."),
        "n_subdominios": 0 if es_ip else max(len(partes) - 2, 0),
        "n_guiones_dominio": host.count("-"),
        "n_digitos": n_dig,
        "ratio_digitos": n_dig / n,
        "n_especiales": sum(not c.isalnum() for c in u),
        "n_arroba": u.count("@"),
        "n_parametros": p.query.count("="),
        "profundidad_path": path.count("/"),
        "tiene_ip": int(es_ip),
        "usa_https": int(low.startswith("https")),
        "tiene_puerto": tiene_puerto,
        "tld_sospechoso": int(tld in TLD_SOSPECHOSOS),
        "es_acortador": int(host in ACORTADORES),
        "n_palabras_sospechosas": sum(w in low for w in PALABRAS_URL),
        "entropia": _entropia(host + path),
        "punycode": int("xn--" in host),
        "doble_slash_path": int("//" in path),
    }


def url_features_df(urls):
    return pd.DataFrame([url_features(u) for u in urls])


def text_indicators(texto):
    t = _norm(texto)
    out = {}
    for nombre, palabras in INDICADORES_TEXTO.items():
        hallados = [p for p in palabras if p in t]
        if hallados:
            out[nombre] = hallados
    return out


def url_indicators(url):
    f = url_features(url)
    razones = []
    if f["tiene_ip"]:
        razones.append("usa una dirección IP en lugar de un dominio")
    if not f["usa_https"]:
        razones.append("no usa HTTPS")
    if f["n_arroba"]:
        razones.append("contiene el símbolo @")
    if f["n_subdominios"] >= 3:
        razones.append("tiene muchos subdominios")
    if f["es_acortador"]:
        razones.append("es un acortador de enlaces")
    if f["tld_sospechoso"]:
        razones.append("usa una terminación de dominio poco habitual")
    if f["n_palabras_sospechosas"] >= 2:
        razones.append("contiene varias palabras típicas de fraude (login, verifica, banco...)")
    if f["longitud"] > 75:
        razones.append("es muy larga")
    if f["punycode"]:
        razones.append("usa caracteres camuflados (punycode)")
    if f["n_guiones_dominio"] >= 2:
        razones.append("el dominio tiene muchos guiones")
    return razones

class FusionPonderada:
    """Fusión sin aprendizaje: usa probabilidades [texto, url, tiene_url].
    La probabilidad no se recalibra al cambiar el umbral; aplicar .predict()
    o comparar predict_proba(... )[:, 1] con .umbral.
    """
    def __init__(self, peso_texto=1.0, umbral=0.5):
        self.peso_texto = float(peso_texto)
        self.umbral = float(umbral)

    def predict_proba(self, X):
        import numpy as np
        x = np.asarray(X, dtype=float)
        if x.ndim != 2 or x.shape[1] != 3:
            raise ValueError('Se necesitan 3 columnas: prob_texto, prob_url, tiene_url')
        pt = np.clip(x[:, 0], 0, 1)
        pu = np.clip(x[:, 1], 0, 1)
        tiene = x[:, 2] > 0.5
        p = np.where(tiene, self.peso_texto * pt +
                     (1 - self.peso_texto) * pu, pt)
        return np.column_stack((1 - p, p))

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= self.umbral).astype(int)
