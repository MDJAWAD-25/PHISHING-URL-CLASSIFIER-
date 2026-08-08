"""
Feature extractor for URLs. Designed to be lightweight and to lazy-import pandas/tldextract
so it remains test-friendly in restricted environments.
"""
from urllib.parse import urlparse
import re

IPV4_RE = re.compile(r"^(?:\d{1,3}\.){3}\d{1,3}$")


def _is_ip(hostname: str) -> bool:
    return bool(IPV4_RE.match(hostname))


def extract_features(url: str) -> dict:
    """Extract a small set of lexical features from a single URL string.

    Returns a dict of feature_name -> value.
    """
    parsed = urlparse(url)
    host = parsed.netloc or parsed.path
    path = parsed.path or ""
    query = parsed.query or ""

    # Basic counts and flags
    features = {}
    features["url_length"] = len(url)
    features["host_length"] = len(host)
    features["path_length"] = len(path)
    features["query_length"] = len(query)
    features["count_dots"] = host.count('.')
    features["count_hyphens"] = host.count('-')
    features["count_digits"] = sum(c.isdigit() for c in url)
    features["has_at"] = 1 if '@' in url else 0
    features["has_double_slash"] = 1 if '//' in urlparse(url).path else 0
    features["starts_with_https"] = 1 if url.lower().startswith('https') else 0
    features["has_ip_in_host"] = 1 if _is_ip(host.split(':')[0]) else 0

    # suspicious token "http" present in hostname (trick used by phishers)
    features["http_in_host"] = 1 if 'http' in host.lower() else 0

    # TLD extraction (lazy import to avoid heavy dependency at module import time)
    try:
        import tldextract

        ext = tldextract.extract(host)
        features["tld"] = ext.suffix or ""
        features["registered_domain"] = ext.registered_domain or host
    except Exception:
        features["tld"] = ""
        features["registered_domain"] = host

    return features


def extract_features_df(urls):
    """Given an iterable of URLs, return a pandas DataFrame of features.

    This does a lazy import of pandas to avoid import-time failures in constrained envs.
    """
    try:
        import pandas as pd
    except Exception as e:
        raise RuntimeError("pandas is required to use extract_features_df: " + str(e))

    rows = [extract_features(u) for u in urls]
    df = pd.DataFrame(rows)
    # Drop non-numeric columns that may have slipped in
    for c in df.columns:
        if df[c].dtype == object:
            try:
                df[c] = pd.to_numeric(df[c], errors='coerce')
            except Exception:
                df = df.drop(columns=[c])
    return df
