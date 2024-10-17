import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, classification_report
# load data
data = pd.read_csv("D:/Projects/365DataScience/intro-to-nlp-for-ai/fake-news-detection-nlp/data/preprocessed_fake_news_data.csv")

X = [','.join(map(str, l)) for l in data['text_clean']]
Y = data['fake_or_factual']

# text vectorization - CountVectorizer
countvec = CountVectorizer()
countvec_fit = countvec.fit_transform(X)
bag_of_words = pd.DataFrame(countvec_fit.toarray(), columns = countvec.get_feature_names_out())

# split into train and test data
X_train, X_test, y_train, y_test = train_test_split(bag_of_words, Y, test_size=0.3)

lr = LogisticRegression(random_state=0).fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)

print("The accuracy for Logistic Regression model is", accuracy_score(y_pred_lr, y_test))

print("Classification Report of SGD model is", classification_report(y_test, y_pred_lr))

svm = SGDClassifier().fit(X_train, y_train)

y_pred_svm = svm.predict(X_test)

print("The accuracy of SGD model is", accuracy_score(y_pred_svm, y_test))

print("Classification Report of SGD model is", classification_report(y_test, y_pred_svm))