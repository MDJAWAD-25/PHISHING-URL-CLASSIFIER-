"""Simple ingestion script: reads CSV with columns (url,label), extracts features, and writes processed CSV and train/test split.
"""
import argparse
import os
import csv

from src.features.feature_extractor import extract_features, extract_features_df


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def read_input_csv(path):
    rows = []
    with open(path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            if 'url' in r:
                rows.append({'url': r['url'], 'label': r.get('label', '')})
    return rows


def write_processed(df, out_path):
    import pandas as pd
    df.to_csv(out_path, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=False, help='Input CSV with url,label')
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()

    out_dir = args.output_dir
    ensure_dir(out_dir)

    # If no input provided, create a tiny sample
    if not args.input:
        sample_path = os.path.join(out_dir, 'sample_malicious_urls.csv')
        with open(sample_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['url', 'label'])
            writer.writerow(['https://github.com', 'benign'])
            writer.writerow(['http://192.168.0.1/login', 'malware'])
            writer.writerow(['http://example.com@evil.com/login', 'phishing'])
        input_path = sample_path
    else:
        input_path = args.input

    rows = read_input_csv(input_path)
    urls = [r['url'] for r in rows]
    labels = [r['label'] for r in rows]

    try:
        import pandas as pd
    except Exception:
        raise RuntimeError('pandas is required to run ingest. Install requirements and retry.')

    features_df = extract_features_df(urls)
    features_df['label'] = labels

    processed_path = os.path.join(out_dir, 'processed.csv')
    write_processed(features_df, processed_path)

    # simple train/test split
    from sklearn.model_selection import train_test_split
    train, test = train_test_split(features_df, test_size=0.4, stratify=features_df['label'] if 'label' in features_df else None, random_state=42)
    train.to_csv(os.path.join(out_dir, 'train.csv'), index=False)
    test.to_csv(os.path.join(out_dir, 'test.csv'), index=False)

    print('Wrote:', processed_path, os.path.join(out_dir, 'train.csv'), os.path.join(out_dir, 'test.csv'))


if __name__ == '__main__':
    main()
