from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV

class PushModel:
    def __init__(
        self,
        classifier_parametrisation: dict,
        calibration_parametrisation: dict,
        prediction_threshold: int,
    ) -> None:
        """
        Args:
            classifier_parametrisation: {
                "GradientBoostingTree parameter": value,
                "GradientBoostingTree parameter2": value,
                ...
            }
            calibration_parametrisation: {
                "CalibratedClassifierCV parameter": value,
                "CalibratedClassifierCV parameter2": value,
                ...
            }
            prediction_threshold: probability threshold above which a prediction is
                considered as 1.
        """

        self.clf = CalibratedClassifierCV(
            GradientBoostingClassifier(**classifier_parametrisation),
            **calibration_parametrisation
        )

        self.prediction_threshold = prediction_threshold