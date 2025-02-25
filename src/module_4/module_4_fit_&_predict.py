import os
import sys
os.chdir('/home/miguel/zrive-ds')
sys.path.append(os.getcwd())

import pandas as pd
import joblib
from src.module_3.metrics_fun import save_model
from src.module_4.push_model import *

def load_training_feature_frame(file_path: str = "training_data.csv") -> pd.DataFrame:
    """
    Carga el conjunto de datos de entrenamiento desde un archivo CSV o una fuente de datos.

    Args:
        file_path (str): Ruta del archivo CSV con los datos de entrenamiento.

    Returns:
        pd.DataFrame: DataFrame con las características para entrenar el modelo.
    """
    try:
        df = pd.read_csv(file_path) 
        return df
    except FileNotFoundError:
        raise Exception(f"Error: El archivo '{file_path}' no fue encontrado.")
    except Exception as e:
        raise Exception(f"Error al cargar los datos: {str(e)}")


def _extract_model_parameters(event: dict) -> tuple[str, dict, dict, float]:
    """
    Extrae los parámetros del modelo desde el evento recibido.

    Args:
        event (Dict): Diccionario con los datos de entrada.

    Returns:
        Tuple[str, Dict, float]: Contiene la ruta del modelo, parámetros del clasificador, y umbral de predicción.
    """
    
    model_folder_path = event.get("model_folder_path", "default_path/")  # Ruta por defecto si no está presente
    
    classifier_parametrisation = event.get("classifier_parametrisation", {})  # Diccionario de configuración del clasificador

    calibration_parametrisation = event.get("calibration_parametrisation", {})  # Diccionario con parámetros de calibración
    
    prediction_threshold = event.get("prediction_threshold", 0.5)  # Valor por defecto si no está en el evento
    
    return model_folder_path, classifier_parametrisation, calibration_parametrisation, prediction_threshold

def classifier_type(classifier_parametrisation: dict) -> str:
    """
    Identifica el tipo de clasificador basado en los parámetros proporcionados.

    Args:
        classifier_parametrisation (dict): Diccionario con los hiperparámetros del clasificador.

    Returns:
        str: Nombre del clasificador ("RandomForest", "XGBoost" o "UnknownModel" si no es uno de los anteriores).
    """ 
    if "n_estimators" in classifier_parametrisation and "criterion" in classifier_parametrisation:
        classifier_name = "RandomForest"
    elif "learning_rate" in classifier_parametrisation and "booster" in classifier_parametrisation:
        classifier_name = "XGBoost"
    else:
        classifier_name = "UnknownModel"  # Fallback en caso de no identificarlo
    
    return classifier_name


def handler_fit(event: dict, _) -> dict[str, any]:
    (
        model_folder_path,
        classifier_parametrisation,
        calibration_parametrisation,
        prediction_threshold,
    ) = _extract_model_parameters(event)

    df = load_training_feature_frame()
    model_name = classifier_type(event)

    clf = PushModel(
        classifier_parametrisation, calibration_parametrisation, prediction_threshold
    )

    clf.fit(df)

    try:
        model_stored_path = save_model(
            clf, model_name, output_path='models/module_4'
        )
    except FileExistsError as e:
        return {"statusCode": "500", "body": {"message": str(e)}}

    return {
        "statusCode": "200",
        "body": {"model_path": str(model_stored_path)}
    }

def handler_predict(event: dict, _) -> dict[str, any]:
    """
    Punto de entrada para la función de predicción.

    Args:
        event: {
            "users": {
                "user_id1": {"feature 1": feature value for id1,
                             "feature 2": feature value for id1, ...},
                "user_id2": {"feature 1": feature value for id2,
                             "feature 2": feature value for id2, ...},
                ...
            },
            "model_path": value
            "prediction_threshold": value
        }
    """

    data_to_predict = pd.DataFrame.from_dict(event["users"], orient="index")
    model_path = event.get("model_path", None)
    clf = joblib.load(model_path)

    predictions = clf.predict(data_to_predict)

    probabilities = clf.predict_proba(data_to_predict)[:, 1]
    predictions = (probabilities >= event.get("prediction_threshold")).astype(int)

    return {
        "statusCode": "200",
        "body": {"prediction": predictions.to_dict()}
    }
