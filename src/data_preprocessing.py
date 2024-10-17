import os
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer


plt.rcParams['figure.figsize'] = (12, 8)
default_plot_colour = "#00bfbf"

# loading the data from .csv file
data = pd.read_csv("D:/Projects/365DataScience/intro-to-nlp-for-ai/fake-news-detection-nlp/data/fake_news_data.csv")

# plot number of fake and factual articles
data['fake_or_factual'].value_counts().plot(kind='bar', color=default_plot_colour)
plt.title('Count of Article Classification')
plt.ylabel('No. of Articles')
plt.xlabel('Classification')
plt.show(block=False)

# a lot of the factual news has a location tag at the beginning of the article, let's use regex to remove this
data['text_clean'] = data.apply(lambda x: re.sub(r"^[^-]*-\s*", "", x['text']), axis=1)

# lowercase 
data['text_clean'] = data['text_clean'].str.lower()

# remove punctuation
data['text_clean'] = data.apply(lambda x: re.sub(r"([^\w\s])", "", x['text_clean']), axis=1)

# stop words
en_stopwords = stopwords.words('english')

data['text_clean'] = data['text_clean'].apply(lambda x: ' '.join([word for word in x.split() if word not in (en_stopwords)]))

# tokenize
data['text_clean'] = data.apply(lambda x: word_tokenize(x['text_clean']), axis=1)

# lemmatize
lemmatizer = WordNetLemmatizer()
data["text_clean"] = data["text_clean"].apply(lambda tokens: [lemmatizer.lemmatize(token) for token in tokens])

folder = "D:/Projects/365DataScience/intro-to-nlp-for-ai/fake-news-detection-nlp/data/"
# Ensure the folder exists
if not os.path.exists(folder):
    os.makedirs(folder)

file_path = os.path.join(folder, "preprocessed_fake_news_data.csv")
data.to_csv(file_path, index=False)

# most common unigrams after preprocessing
tokens_clean = sum(data['text_clean'], [])
unigrams = pd.Series(nltk.ngrams(tokens_clean, 1)).value_counts()
unigrams.index = [' '.join(gram) for gram in unigrams.index]
print("Most common Unigrams after pre-processsing", unigrams[:10])

sns.barplot(x = unigrams.values[:10], 
            y = unigrams.index[:10], 
            orient = 'h',
            palette=[default_plot_colour])\
.set(title='Most Common Unigrams After Preprocessing')

plt.show(block=False)

# most common bigrams after preprocessing
bigrams = (pd.Series(nltk.ngrams(tokens_clean, 2)).value_counts()) 
print("Most common bigrams after preprocessing",bigrams[:10])