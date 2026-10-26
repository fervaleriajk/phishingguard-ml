"""Inferencia segura sin llamadas de red para los modelos guardados por Colab.

Convención: 1 = phishing, 0 = correo o URL legítima.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from features import extract_urls, text_indicators, url_features_df, url_indicators

MAX_URLS = 30


@dataclass
class Resultado:
    clase: int
    score: float
    umbral: float
    score_texto: float | None
    score_url: float | None
    urls: list[str]
    senales_texto: dict[str, list[str]]
    senales_urls: dict[str, list[str]]


def obtener_urls(texto: str, enlaces_adicionales: str = "") -> list[str]:
    """Detecta enlaces existentes o pegados por separado, sin abrirlos."""
    enlaces = extract_urls(texto) + extract_urls(enlaces_adicionales)
    for linea in str(enlaces_adicionales).splitlines():
        candidato = linea.strip().strip("'\"[],;")
        if re.fullmatch(r"(?:[\w-]+\.)+[a-zA-Z]{2,}(?::\d{1,5})?(?:[/#?]\S*)?", candidato):
            enlaces.append(candidato)
    return list(dict.fromkeys(enlaces))[:MAX_URLS]


def puntuar_urls(modelo_url: Any, urls: list[str]) -> list[float]:
    if not urls:
        return []
    probabilidades = modelo_url.predict_proba(url_features_df(urls))
    return [float(v) for v in probabilidades[:, 1]]


def analizar_correo(modelo_texto: Any, modelo_url: Any, modelo_fusion: Any,
                   asunto: str, cuerpo: str, enlaces_adicionales: str = "") -> Resultado:
    texto = f"{asunto.strip()} {cuerpo.strip()}".strip()
    if not texto:
        raise ValueError("Escribe un asunto o cuerpo de correo para analizar.")
    urls = obtener_urls(texto, enlaces_adicionales)
    prob_texto = float(modelo_texto.predict_proba([texto])[0, 1])
    probabilidades_url = puntuar_urls(modelo_url, urls)
    prob_url = max(probabilidades_url, default=0.0)
    x = np.array([[prob_texto, prob_url, float(bool(urls))]], dtype=float)
    score = float(modelo_fusion.predict_proba(x)[0, 1])
    umbral = float(modelo_fusion.umbral)
    return Resultado(
        clase=int(score >= umbral),
        score=score,
        umbral=umbral,
        score_texto=prob_texto,
        score_url=prob_url if urls else None,
        urls=urls,
        senales_texto=text_indicators(texto),
        senales_urls={url: url_indicators(url) for url in urls},
    )


def analizar_url(modelo_url: Any, url: str, umbral: float = 0.50) -> Resultado:
    enlace = url.strip()
    if not enlace or any(c.isspace() for c in enlace):
        raise ValueError("Introduce una única URL válida (sin espacios).")
    valor = puntuar_urls(modelo_url, [enlace])[0]
    return Resultado(
        clase=int(valor >= umbral),
        score=valor,
        umbral=umbral,
        score_texto=None,
        score_url=valor,
        urls=[enlace],
        senales_texto={},
        senales_urls={enlace: url_indicators(enlace)},
    )
