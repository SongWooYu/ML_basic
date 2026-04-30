from sklearn.datasets import load_breast_cancer
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

data = load_breast_cancer(as_frame=True)
# print(data.frame)

# print(data.data)
# print(data.target)

# pairplot
import seaborn as sns
sns.pairplot(data.frame, hue='target')

X_train, X_test, y_train, y_test = train_test_split(data.data.iloc[:, :1], data.target, test_size=0.3, random_state=1234)
# print(X_train.shape)
# print(X_test.shape)
# print(y_train.shape)
# print(y_test.shape)

X_train, X_test, y_train, y_test = train_test_split(data.data.iloc[:, :1], data.target, test_size=0.3, random_state=1234)
# print(X_train.shape)
# print(X_test.shape)
# print(y_train.shape)
# print(y_test.shape)

X_test_std = scalerX.transform(X_test)
# print(X_test_std)

clf = svm.SVC(kernel='linear')
clf.fit(X_train_std, y_train)
y_pred = clf.predict(X_test_std)
# print(y_pred)

cf = confusion_matrix(y_test, y_pred)
# print(cf)
clf.score(X_test_std, y_test)

s = 0
e = 1

X_train, X_test, y_train, y_test = train_test_split(data.data.iloc[:, s:e], data.target, test_size=0.3, random_state=1234)
# print(X_train.shape)
# print(X_test.shape)
# print(y_train.shape)
# print(y_test.shape)
scalerX = StandardScaler()
scalerX.fit(X_train)
X_train_std = scalerX.transform(X_train)
# print(X_train_std)
X_test_std = scalerX.transform(X_test)
# print(X_test_std)
clf = svm.SVC(kernel='linear')
clf.fit(X_train_std, y_train)
y_pred = clf.predict(X_test_std)
# print(y_pred)
cf = confusion_matrix(y_test, y_pred)
# print(cf)
clf.score(X_test_std, y_test)