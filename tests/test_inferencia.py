"""Pruebas de inferencia, sin subir ni descargar modelos reales."""
import numpy as np

from features import FusionPonderada
from inferencia import analizar_correo, analizar_url, obtener_urls


class ModeloTextoFalso:
    def predict_proba(self, entrada):
        return np.tile(np.array([[0.2, 0.8]]), (len(entrada), 1))


class ModeloURLFalso:
    def predict_proba(self, entrada):
        return np.tile(np.array([[0.75, 0.25]]), (len(entrada), 1))


def test_convencion_y_umbral_entrenado():
    r = analizar_correo(ModeloTextoFalso(), ModeloURLFalso(), FusionPonderada(.8, .65),
                       'Actualización', 'Consultar https://example.org/a')
    assert r.clase == 1
    assert abs(r.score - .69) < 1e-10
    assert r.umbral == .65
    assert r.urls == ['https://example.org/a']


def test_correo_sin_url_solo_texto():
    r = analizar_correo(ModeloTextoFalso(), ModeloURLFalso(), FusionPonderada(.2, .85),
                       'Boletín', 'Reunión semanal')
    assert r.score == .8
    assert r.clase == 0
    assert r.score_url is None


def test_url_individual_1_phishing_0_legitimo():
    r = analizar_url(ModeloURLFalso(), 'www.example.org')
    assert r.clase == 0
    assert r.score == .25
    assert r.umbral == .5


def test_extraer_urls_sin_duplicados():
    r = obtener_urls('Ver https://example.org/r', 'https://example.org/r\nexample.com/login')
    assert r == ['https://example.org/r', 'example.com/login']
