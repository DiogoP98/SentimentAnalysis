# Sentiment Analysis

This project analyses the performance of different methods in the challenging task of sentiment analysis. In order to do so, Amazon Kindle data was obtained from Kaggle (https://www.kaggle.com/bharadwaj6/kindle-reviews).


## Implementation

The implementation process consisted in four main steps:
* Pre-processing: Sampling (to overcome the class imbalance issue) and text cleaning.
* Exploratory Data Analysis (EDA): N-grams, word count, etc.
* Classification: Used both classical machine learning methods (such as Gradient Boosting, SVMs and Gaussian Naive Bayes) and deep learning methods (e.g. BERT, RoBERTa, LSTMs)
* Text generation

### Enviroment

* Python 3.6+
* Pandas
* Spacy
* Tqdm
* Spacymoji
* Numpy
* Sklearn
* Imblearn
* Matplotlib
* Seaborn
* Gensim
* PyLDAvis
* Logging
* Nltk
* Wordcloud
* Torchbearer
* PyTorch
* Transformers

Install all python modules with

```
pip install -r requirements.txt
```
or if you have different versions of Python installed:
```
pip3 install -r requirements.txt
```

## Authors

* **Alex Newton** - [xandernewton](https://github.com/xandernewton)

* **Diogo Pereira** - [DiogoP98](https://github.com/DiogoP98)




