from analyzers.sentiment_parser import SentimentParser

class DummySentimentModel:
    def __call__(self, texts):
        return [{'label': 'positive'} if 'good' in t else {'label': 'negative'} if 'bad' in t else {'label': 'neutral'} for t in texts]

def test_classify_sentiment():
    sp = SentimentParser()
    sp.sentiment_model = DummySentimentModel()
    assert sp.classify_sentiment(['good news']) == 1
    assert sp.classify_sentiment(['bad news']) == -1
    assert sp.classify_sentiment(['neutral news']) == 0
    assert sp.classify_sentiment(['good news', 'bad news']) == 0 