import pandas as pd
import seaborn as sns
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# use vader so we also get a neutral sentiment count
vader_sentiment = SentimentIntensityAnalyzer()

# load data
data = pd.read_csv("D:/Projects/365DataScience/intro-to-nlp-for-ai/fake-news-detection-nlp/data/preprocessed_fake_news_data.csv")

data['vader_sentiment_score'] = data['text'].apply(lambda review: vader_sentiment.polarity_scores(review)['compound'])

# create labels
bins = [-1, -0.1, 0.1, 1]
names = ['negative', 'neutral', 'positive']

data['vader_sentiment_label'] = pd.cut(data['vader_sentiment_score'], bins, labels=names)

data['vader_sentiment_label'].value_counts().plot.bar()

sns.countplot(
    x = 'fake_or_factual',
    hue = 'vader_sentiment_label',
    palette = sns.color_palette("hls"),
    data = data
) \
.set(title='Sentiment by News Type')