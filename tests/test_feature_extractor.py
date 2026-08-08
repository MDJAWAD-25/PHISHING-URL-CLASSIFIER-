from src.features.feature_extractor import extract_features, extract_features_df


def test_extract_single():
    url = 'https://github.com'
    f = extract_features(url)
    assert isinstance(f, dict)
    assert f['url_length'] > 0
    assert 'host_length' in f


def test_extract_df():
    urls = ['https://github.com', 'http://example.com/login']
    df = extract_features_df(urls)
    assert df.shape[0] == 2
    assert 'url_length' in df.columns
