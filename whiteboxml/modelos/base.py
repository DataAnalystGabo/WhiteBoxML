"""
Módulo con la abstracción base para los modelos discriminantes.

:authors: Gonzalo Ramirez
:date: 09/06/2026
"""

# EXPLICACIÓN PARA EL EQUIPO:
# El issue pide una "abstracción conjunta que ahorre código".
# Tanto LDA como QDA necesitan calcular:
# 1. Las probabilidades previas (priors) de cada clase.
# 2. Los vectores de medias (centroides) de cada clase.
#
# TAREA:
# Crear aquí la clase `BaseDiscriminantAnalysis`.
# Implementar un método `_compute_priors_and_means(self, X, y)` que resuelva
# esos cálculos iniciales. Luego, LDA y QDA heredarán esta clase y llamarán
# a este método dentro de su propio `fit()`.
