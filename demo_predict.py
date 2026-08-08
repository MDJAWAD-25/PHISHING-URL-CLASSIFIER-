"""Demo CLI to predict a single URL using the trained RandomForest artifact.
Usage: python demo_predict.py "http://example.com"
"""
import sys
import os
import pickle

from src.features.feature_extractor import extract_features, extract_features_df

MODEL_PATH = os.path.join('src', 'models', 'artifacts', 'randomforest.pkl')


def main():
    if len(sys.argv) < 2:
        print('Usage: python demo_predict.py "http://example.com"')
        return
    url = sys.argv[1]
    if not os.path.exists(MODEL_PATH):
        print('Model artifact not found. Run: python -m src.models.train --data data/train.csv --output-dir src/models/artifacts')
        return
    with open(MODEL_PATH, 'rb') as f:
        art = pickle.load(f)
    feats = extract_features(url)
    import pandas as pd
    X = pd.DataFrame([feats])[art['feature_names']].fillna(0)
    pred_idx = int(art['model'].predict(X)[0])
    label = art['label_encoder'].inverse_transform([pred_idx])[0]
    print('Predicted class:', label)


if __name__ == '__main__':
    main()
