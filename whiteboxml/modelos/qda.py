"""
Implementación del Análisis Discriminante Cuadrático (QDA).

:authors: Gonzalo Ramirez
:date: 09/06/2026
"""

# EXPLICACIÓN PARA EL EQUIPO:
# TAREA PRINCIPAL (Compañeros):
# 1. Importar la clase base: `from .base import BaseDiscriminantAnalysis`

#--------- Camabios en el código de LDA.py para QDA.py ---------
from typing import Optional, cast
import numpy as np
from numpy.typing import ArrayLike
from whiteboxml.modelos.base import BaseDiscriminantAnalysis
#--------- ---------

# 2. Crear la clase `QDA(BaseDiscriminantAnalysis)`
class QDA(BaseDiscriminantAnalysis):
    """
    Clasificador por Análisis Discriminante Cuadrático (QDA).

    Instancia el modelo asumiendo covarianza distinta por clase.

    :authors: Gonzalo Ramirez
    :date: 09/06/2026
    """

    def __init__(self) -> None:
        """
        Inicializa el modelo QDA configurando la covarianza como nula.

        :authors: Gonzalo Ramirez
        :date: 09/06/2026
        """
        super().__init__()
        self.covariances_: Optional[np.ndarray] = None  # Almacena una matriz de covarianza por clase por eso es plural     
    
    def fit(self, features: ArrayLike, targets: ArrayLike) -> "QDA":
        """
        Ajusta el modelo QDA calculando la covarianza para cada clase.

        :param features: Matriz de características de entrenamiento.
        :type features: ArrayLike
        :param targets: Vector de etiquetas reales.
        :type targets: ArrayLike
        :return: La propia instancia entrenada.
        :rtype: QDA
        :authors: Gonzalo Ramirez
        :date: 09/06/2026
        """
        self._compute_priors_and_means(features, targets)

        if self.classes_ is None or self.means_ is None:
            raise ValueError("Falló el cálculo de estadísticas base.")

        valid_classes = cast(np.ndarray, self.classes_)
        valid_means = cast(np.ndarray, self.means_)

        features_array = np.asarray(features)
        targets_array = np.asarray(targets)

        n_samples = features_array.shape[0]
        n_features = features_array.shape[1]

        # Calcular una matriz de covarianza para cada clase
        self.covariances_ = np.zeros((len(valid_classes), n_features, n_features))
        
        for idx, cls in enumerate(valid_classes):
            class_features = features_array[targets_array == cls]
            centered_features = class_features - valid_means[idx]
            self.covariances_[idx] = np.dot(centered_features.T, centered_features) / (class_features.shape[0] - 1)

        return self
    def predict(self, features: ArrayLike) -> np.ndarray:
        """
        Predice las etiquetas para las muestras dadas utilizando la función de score cuadrático.

        :param features: Matriz de características de prueba.
        :type features: ArrayLike
        :return: Vector de etiquetas predichas.
        :rtype: np.ndarray
        :authors: Gonzalo Ramirez
        :date: 09/06/2026
        """
        if self.classes_ is None or self.means_ is None or self.covariances_ is None or self.priors_ is None:# se agrega la verificación de covarianzas
            raise ValueError("El modelo no ha sido entrenado correctamente.")

        valid_classes = cast(np.ndarray, self.classes_)
        valid_means = cast(np.ndarray, self.means_)
        covariances = cast(np.ndarray, self.covariances_)

        features_array = np.asarray(features)
        n_samples = features_array.shape[0]

        scores = np.zeros((n_samples, len(valid_classes)))

        for idx, cls in enumerate(valid_classes):
            mean_vec = valid_means[idx]
            cov_matrix = covariances[idx]
            inv_cov_matrix = np.linalg.inv(cov_matrix)
            det_cov_matrix = np.linalg.det(cov_matrix)

            for i in range(n_samples):
                diff = features_array[i] - mean_vec
                scores[i, idx] = -0.5 * np.dot(diff.T, np.dot(inv_cov_matrix, diff)) - 0.5 * np.log(det_cov_matrix) + np.log(self.priors_[idx])

        predicted_indices = np.argmax(scores, axis=1)
        return valid_classes[predicted_indices]


# 3. En el método `fit(X, y)`:
#    - Llamar a `self._compute_priors_and_means(X, y)`
#    - Calcular una MATRIZ DE COVARIANZA INDIVIDUAL para CADA clase.
# 4. En el método `predict(X)`:
#    - Implementar la función de score cuadrático. A diferencia de LDA,
#      acá los términos cuadráticos no se cancelan porque las matrices
#      de covarianza son distintas por clase.
