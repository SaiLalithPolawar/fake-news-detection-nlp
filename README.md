# Fake News Detection with NLP

## Case Description
Imagine you are working for a social media company. The company is concerned with the growing amount of fake news circulating on its platform. They have assigned you to investigate how fake news can be recognized and create a method for identifying it.

This project involves:

- Exploring and cleaning textual data.
- Classifying fake vs factual news stories using machine learning and NLP techniques.
- Performing sentiment analysis to gauge the tone of news articles.
- Visualizing results to communicate findings effectively to stakeholders.

## Project Structure

````bash
/fake-news-detection-nlp
|
|-- /data
|   |-- fake_news_data.csv               # The dataset containing news articles and their labels
│   |-- preprocessed_fake_news_data.csv  # preprocessed and clean data
|
|-- /notebooks
|   |-- fake_news_detection.ipynb        # Main exploratory notebook combining all steps
|
|-- /src
|   |-- data_preprocessing.py            # Script for text cleaning and preprocessing
|   |-- sentiment_analysis.py            # Script for sentiment analysis using SentimentIntensityAnalyzer
|   |-- topic_modelling.py               # Script for feature extraction (e.g., TF-IDF, count vectors)
|   |-- model_training.py                # Script for training and evaluating models
|   |-- utils.py                         # helper methods
|
|-- requirements.txt                     # List of dependencies
|-- README.md                            # This file
|-- .gitignore                           # Files to ignore in version control

````

## How to Run the Project
1. Clone the repository:
```` bash
git clone https://github.com/yourusername/NLP-FakeNews-Detection.git
cd NLP-FakeNews-Detection
````
2. Install dependencies: Make sure you have Python installed. You can install the required libraries using:
````bash
pip install -r requirements.txt
````
3. Prepare the dataset:
- Place your data folder
4. Run the Jupyter Notebook:
- Open the notebook ```notebooks/fake_news_detection.ipynb``` and follow the step-by-step instructions to explore, clean, and analyze the data.
5. Run scripts separately (optional):
- You can also run each module independently:
  - Preprocess the data using ````data_preprocessing.py````
  - Implement semantic analysis ````semantic_analysis.py````
  - Extract features using ````topic_modelling.py````
  - Train and evaluate models using ````model_training.py````

## Key Components
1. Data Preprocessing (src/data_preprocessing.py):
- Handles text cleaning: tokenization, stopword removal, stemming/lemmatization, etc.
- Prepares the data for further analysis and modeling.

2. Topic Modeling (src/feature_extraction.py):
- Extracts text features using techniques like TF-IDF and count vectorization to transform text into numerical data for model training.

3. Sentiment Analysis (src/sentiment_analysis.py):
- Performs sentiment analysis  over the data and classifies the news into positive, neutral and negative emotions.

4. Model Training (src/model_training.py):
- Trains machine learning models such as Logistic Regression or SGDClassifier to classify news as fake or factual.
- Evaluates the model using metrics like accuracy, classification reports, and confusion matrices.

5. Exploratory Notebook (notebooks/fake_news_detection.ipynb):
- Combines all steps in one place for exploratory analysis.
- Visualizes results using matplotlib and seaborn.

## Results
The project aims to:
- Accurately classify fake and factual news using NLP and machine learning techniques.
- Provide visualizations to better understand model performance and the distribution of fake news in the dataset.