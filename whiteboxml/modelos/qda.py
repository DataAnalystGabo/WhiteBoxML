"""
Implementación del Análisis Discriminante Cuadrático (QDA).

:authors: Gonzalo Ramirez
:date: 09/06/2026
"""

# EXPLICACIÓN PARA EL EQUIPO:
# TAREA PRINCIPAL (Compañeros):
# 1. Importar la clase base: `from .base import BaseDiscriminantAnalysis`
# 2. Crear la clase `QDA(BaseDiscriminantAnalysis)`
# 3. En el método `fit(X, y)`:
#    - Llamar a `self._compute_priors_and_means(X, y)`
#    - Calcular una MATRIZ DE COVARIANZA INDIVIDUAL para CADA clase.
# 4. En el método `predict(X)`:
#    - Implementar la función de score cuadrático. A diferencia de LDA,
#      acá los términos cuadráticos no se cancelan porque las matrices
#      de covarianza son distintas por clase.
