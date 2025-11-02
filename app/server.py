# app/server.py
import mlflow
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional

# ---- Hard-coded config (simple, explicit) ----
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
MODEL_NAME          = "iris-classifier"
DEFAULT_VERSION       = "1"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


# Dictionary to hold loaded models by version
models = {}
current_version = DEFAULT_VERSION

def load_model_version(version: str):
    """Load a specific version from MLflow model registry."""
    model_uri = f"models:/{MODEL_NAME}/{version}"
    try:
        loaded_model = mlflow.pyfunc.load_model(model_uri)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Model version {version} not found: {e}")
    models[version] = loaded_model
    return loaded_model

# Load default version at startup
models[DEFAULT_VERSION] = load_model_version(DEFAULT_VERSION)

#MODEL_URI = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
#model = mlflow.pyfunc.load_model(MODEL_URI)

# ----- Pydantic schemas with helpful docs + examples -----
class IrisSample(BaseModel):
    sepal_length: float = Field(..., ge=0, description="Sepal length in cm")
    sepal_width:  float = Field(..., ge=0, description="Sepal width in cm")
    petal_length: float = Field(..., ge=0, description="Petal length in cm")
    petal_width:  float = Field(..., ge=0, description="Petal width in cm")

class PredictRequest(BaseModel):
    samples: List[IrisSample]

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "samples": [
                        {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},
                        {"sepal_length": 6.7, "sepal_width": 3.1, "petal_length": 4.7, "petal_width": 1.5},
                        {"sepal_length": 6.3, "sepal_width": 3.3, "petal_length": 6.0, "petal_width": 2.5}
                    ]
                }
            ]
        }
    }

# For convenience, return both class ids and human labels
IRIS_LABELS = {0: "setosa", 1: "versicolor", 2: "virginica"}

class PredictResponse(BaseModel):
    class_id: List[int]    # 0,1,2
    class_label: List[str] # setosa/versicolor/virginica

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"class_id": [0, 1, 2], "class_label": ["setosa", "versicolor", "virginica"]}
            ]
        }
    }

app = FastAPI(
    title="Iris Classifier API",
    description="Predict Iris species from sepal/petal measurements (cm).",
    version="1.0.0",
)

@app.get("/health", tags=["health"])
def health():
    #return {"status": "ok", "model_uri": MODEL_URI}
    return {
        "status": "ok",
        "current_model_version": current_version,
        "loaded_versions": list(models.keys())
    }

@app.post(
    "/predict",
    response_model=PredictResponse,
    tags=["prediction"],
    summary="Predict Iris species",
    #description="Send one or more Iris samples; returns class id (0,1,2) and label (setosa, versicolor, virginica)."
    description="Send one or more Iris samples; returns predictions using the current or specified model version."

)
# def predict(req: PredictRequest) -> PredictResponse:
#     # TODO Run predict
def predict(req: PredictRequest, version: Optional[str] = None) -> PredictResponse:
    global current_version

    model_version = version or current_version
    if model_version not in models:
        # Lazy-load model if not already loaded
        load_model_version(model_version)

    model = models[model_version]
    
    #Convert request to model input
    # import pandas as pd
    # X = pd.DataFrame({
    #     "sepal_length": req.sepal_length,
    #     "sepal_width": req.sepal_width,
    #     "petal_length": req.petal_length,
    #     "petal_width": req.petal_width
    # })
    
    #predict not working - made change below to correct predict input - chatGPT
    import pandas as pd
    X = pd.DataFrame([s.model_dump() for s in req.samples])

    preds = model.predict(X)

    # Convert numeric predictions to labels if needed
    label_map = {0: "setosa", 1: "versicolor", 2: "virginica"}
    class_id = [int(p) for p in preds]
    class_label = [label_map[i] for i in class_id]

#    return PredictResponse(class_id=class_id, class_label=class_label)


    # return PredictResponse(
    #     class_id=[],
    #     class_label=[]
    # )
    #replaced code above with this code - chatGPT
    return PredictResponse(
    class_id=class_id,
    class_label=class_label
    )

    
    
#Endpoint to register models - allows API to dynamically load and optionally activate new model versions
class RegisterModelRequest(BaseModel):
    version: str  # e.g., "2"
    set_as_current: Optional[bool] = True  # whether to make it active immediately

@app.post("/register_model", tags=["model"])
def register_model(req: RegisterModelRequest):
    global current_version
    if req.version in models:
        raise HTTPException(status_code=400, detail=f"Version {req.version} already loaded")

    load_model_version(req.version)

    if req.set_as_current:
        current_version = req.version

    return {"message": f"Model version {req.version} registered.",
            "current_version": current_version,
            "loaded_versions": list(models.keys())}


# TODO Add endpoint to get the current model serving version

from fastapi import Body

@app.get("/model/version", tags=["model"])
def get_current_version():
    """
    Return the currently served model version.
    """
    return {
        "current_version": current_version,
        "loaded_versions": list(models.keys())
    }

# TODO Add endpoint to update the serving version
@app.post("/model/version", tags=["model"])
def set_current_version(
    version: str = Body(..., embed=True, description="Version number to serve (e.g., '2')")
):
    """
    Update the currently served model version.
    Loads it from MLflow if not already loaded.
    """
    global current_version

    # Load model if not already cached
    if version not in models:
        load_model_version(version)

    # Update the active serving version
    current_version = version

    return {
        "message": f"Now serving model version {version}",
        "current_version": current_version,
        "loaded_versions": list(models.keys())
    }


