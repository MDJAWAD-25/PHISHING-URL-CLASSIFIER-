from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import os

from src.features.feature_extractor import extract_features, extract_features_df

app = FastAPI(title='Malicious URL Classifier (demo)')

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'artifacts', 'randomforest.pkl')
_artifact = None


class URLRequest(BaseModel):
    url: str


def load_model():
    global _artifact
    if _artifact is not None:
        return _artifact
    p = os.path.abspath(MODEL_PATH)
    if not os.path.exists(p):
        return None
    with open(p, 'rb') as f:
        _artifact = pickle.load(f)
    return _artifact


@app.on_event('startup')
def startup():
    load_model()


@app.post('/predict')
def predict(req: URLRequest):
    artifact = load_model()
    features = extract_features(req.url)
    # ensure numeric ordering
    if artifact is None:
        raise HTTPException(status_code=503, detail='model artifact not found; run training first')

    import pandas as pd
    X = pd.DataFrame([features])[artifact['feature_names']].fillna(0)
    probs = artifact['model'].predict_proba(X)[0]
    pred_idx = int(artifact['model'].predict(X)[0])
    label = artifact['label_encoder'].inverse_transform([pred_idx])[0]
    classes = list(artifact['label_encoder'].classes_)

    out = {
        'url': req.url,
        'prediction': label,
        'probabilities': {c: float(probs[i]) for i, c in enumerate(classes)}
    }
    return out


@app.get('/')
def root():
    return {'status': 'ok', 'model_loaded': load_model() is not None}
