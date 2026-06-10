"""
Implementación del Análisis Discriminante Lineal (LDA).

:authors: Gonzalo Ramirez
:date: 09/06/2026
"""

# EXPLICACIÓN:
# TAREA PRINCIPAL (Gonzalo):
# 1. Importar la clase base: `from .base import BaseDiscriminantAnalysis`
# 2. Crear la clase `LDA(BaseDiscriminantAnalysis)`
# 3. En el método `fit(X, y)`:
#    - Llamar a `self._compute_priors_and_means(X, y)`
#    - Calcular la MATRIZ DE COVARIANZA COMPARTIDA para todas las clases.
# 4. En el método `predict(X)`:
#    - Implementar la regla de decisión lineal (score discriminante)
#      utilizando la covarianza compartida.
