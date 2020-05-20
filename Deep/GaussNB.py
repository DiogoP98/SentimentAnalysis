import numpy as np
import sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

df = pd.read_csv("Datasets\\new_clean_2million.csv")
print("Downloaded data of shape:")
print(df.shape)
df = df.dropna()
print("Dropped nans")
print(df.shape)

X = df.reviewText.values
y = df.overall.values
X, y  = X[:1000], y[:1000]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

clf = GaussianNB()
batches = 10
step_size = len(X_train.toarray())/batches
for i in range(batches):
    X_batch = X_train[int(i*step_size) : int((i+1)*step_size) - 1].toarray()
    y_batch = y_train[int(i*step_size) : int((i+1)*step_size) - 1]
    clf.fit(X_batch,y_batch)
batches = 5
score = []
step_size = len(X_test.toarray())/batches

for i in range(batches):
    X_batch = X_test[int(i*step_size) : int((i+1)*step_size) - 1].toarray()
    y_batch = y_test[int(i*step_size) : int((i+1)*step_size) - 1]
    y_pred = clf.predict(X_batch)
    score.append(accuracy_score(y_batch, y_pred))
print(score)
print(np.sum(score)/5)
'''[0.8307692307692308, 0.7076923076923077, 0.7846153846153846, 0.7692307692307693, 0.7538461538461538]
average: 0.7692307692307693'''