import numpy as np
from sklearn.metrics import f1_score
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
#df = pd.read_csv("Datasets\\new_clean_sm.csv")

'''[0.23076923076923078, 0.2153846153846154, 0.23076923076923078, 0.26153846153846155, 0.18461538461538463]
average accuracy score: 0.2246153846153846
f1macro average score: 0.2070097434322828'''
#df = pd.read_csv("Datasets\\new_clean_2million.csv")
'''[0.6615384615384615, 0.7846153846153846, 0.7076923076923077, 0.7230769230769231, 0.6615384615384615]
average accuracy score: 0.7076923076923076
f1macro average score: 0.177219595968726'''

df = pd.read_csv(r'/Users/kai/Documents/Southampton/Datasets/new_clean_sm_100000.csv')
'''[0.26153846153846155, 0.23076923076923078, 0.27692307692307694, 0.27692307692307694, 0.27692307692307694]
average accuracy score: 0.26461538461538464
f1macro average score: 0.2616784391699993'''
print("Downloaded data of shape:")
print(df.shape)
df = df.drop(['vote', 'image'],axis=1)
df = df.dropna()
print("Dropped nans")
these_vals = [1.0, 3.0, 5.0]
df.loc[df['overall'].isin(these_vals)]
print(df.shape)

X = df.reviewText.values
y = df.overall.values
X, y  = X[:1000], y[:1000]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

clf = GaussianNB()
clf.set_params(var_smoothing=2.5e-8)
batches = 10
step_size = len(X_train.toarray())/batches
for i in range(batches):
    print(f'batch {i}/{batches}')
    X_batch = X_train[int(i*step_size) : int((i+1)*step_size) - 1].toarray()
    y_batch = y_train[int(i*step_size) : int((i+1)*step_size) - 1]

    clf.partial_fit(X_batch,y_batch, classes=np.unique(y))

batches = 5
score = []
f =[]
cv_score = []
step_size = len(X_test.toarray())/batches

for i in range(batches):
    X_batch = X_test[int(i*step_size) : int((i+1)*step_size) - 1].toarray()
    y_batch = y_test[int(i*step_size) : int((i+1)*step_size) - 1]
    y_pred = clf.predict(X_batch)
    score.append(accuracy_score(y_batch, y_pred))
    f.append(f1_score(y_batch,y_pred,average='macro'))
    cv_score.append(cross_val_score(clf, X_batch, y_batch))
#print(score)
print(f'average accuracy score: {np.sum(score)/5}')
print(f'f1macro average score: {np.mean(f)}')
print(clf.get_params())
# print('finnie')
print(f'average cv score: {np.mean(cv_score)}')
'''[0.8307692307692308, 0.7076923076923077, 0.7846153846153846, 0.7692307692307693, 0.7538461538461538]
average: 0.7692307692307693'''