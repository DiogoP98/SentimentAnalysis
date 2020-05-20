from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
from sklearn.metrics import f1_score
import sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
#df = pd.read_csv("Datasets\\new_clean_sm.csv")
'''average 0.21538461538461542
finnie
f1macro average score: 0.2079264894643041'''
#df = pd.read_csv("Datasets\\new_clean_2million.csv")
df = pd.read_csv(r'/Users/kai/Documents/Southampton/Datasets/new_clean_sm_100000.csv')
'''[0.5230769230769231, 0.4461538461538462, 0.49230769230769234, 0.35384615384615387, 0.4307692307692308]
0.4492307692307692
finnie
f1macro average score: 0.42892581051650264'''
print("Imported data of shape:")
print(df.shape)
df = df.drop(['vote', 'image'],axis=1)
df = df.dropna()
#print(df.head())
these_vals = [1.0, 3.0, 5.0]
df = df.loc[df['overall'].isin(these_vals)]
print(df.shape)

X = df.reviewText.values
y = df.overall.values
X, y  = X[:1000], y[:1000]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)
f =[]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

clf = GradientBoostingClassifier(warm_start=True)
clf.set_params(learning_rate = 0.05)

batches = 10
step_size = len(X_train.toarray())/batches
for i in range(batches):
    print(f'training batch {i}/{batches}')
    X_batch = X_train[int(i*step_size) : int((i+1)*step_size) - 1].toarray()
    y_batch = y_train[int(i*step_size) : int((i+1)*step_size) - 1]
    clf.fit(X_batch,y_batch)

batches = 5
score = []
step_size = len(X_test.toarray())/batches
cv_score = []
for i in range(batches):
    print(f'testing batch {i}/{batches}')
    X_batch = X_test[int(i*step_size) : int((i+1)*step_size) - 1].toarray()
    y_batch = y_test[int(i*step_size) : int((i+1)*step_size) - 1]
    y_pred = clf.predict(X_batch)
    score.append(accuracy_score(y_batch, y_pred))
    f.append(f1_score(y_batch, y_pred, average='macro'))
    cv_score.append(cross_val_score(clf, X_batch, y_batch))

#print(score)
print(f'average accuracy score: {np.sum(score)/5}')
print(f'f1macro average score: {np.mean(f)}')
print(clf.get_params())
print(f'average cv score: {np.mean(cv_score)}')

print('finnie')
'''[0.8307692307692308, 0.7076923076923077, 0.7846153846153846, 0.7692307692307693, 0.7538461538461538]
average: 0.7692307692307693'''