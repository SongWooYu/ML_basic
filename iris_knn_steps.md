# Iris KNN 분류 — 단계별 가이드

**목표:** `Iris` 데이터셋을 K-최근접 이웃(KNN)으로 분류하고, `pairplot`으로 탐색적 데이터 분석(EDA)을 수행하여 특징별 분포와 클래스 구분을 확인합니다.

**요구사항:**
- Python 3.8+
- 라이브러리: `scikit-learn`, `pandas`, `seaborn`, `matplotlib`, `joblib` (선택)

**단계 요약**

1. **데이터 로드**: `load_iris`로 데이터셋을 불러와 `DataFrame`으로 정리

```python
from sklearn.datasets import load_iris
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = load_iris(as_frame=True)
df = data.frame.copy()
# 편의 컬럼 추가
df['target'] = data.target
mapping = dict(enumerate(data.target_names))
df['target_name'] = df['target'].map(mapping)
```

2. **EDA — Pairplot**: 클래스별 산포와 분포 확인

```python
sns.pairplot(df, vars=data.feature_names, hue='target_name', diag_kind='hist', palette='Set1')
plt.suptitle('Iris Pairplot', y=1.02)
plt.show()
```

- `pairplot`으로 어떤 피처 쌍이 클래스 구분에 유리한지 시각적으로 확인합니다.

3. **데이터 분할 및 표준화**

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)
```

- `stratify=y`로 클래스 비율 유지
- KNN은 거리 기반이므로 표준화 필요

4. **K 값 탐색 및 모델 학습**

```python
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt

neighbors = range(1, 16)
train_acc = []
test_acc = []

for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_std, y_train)
    train_acc.append(knn.score(X_train_std, y_train))
    test_acc.append(knn.score(X_test_std, y_test))

plt.plot(neighbors, train_acc, label='Train')
plt.plot(neighbors, test_acc, label='Test')
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# 최종 k 선택 후 학습
k_best = int(neighbors[np.argmax(test_acc)])
knn_best = KNeighborsClassifier(n_neighbors=k_best)
knn_best.fit(X_train_std, y_train)
```

5. **평가 (혼동행렬, classification report, F1)**

```python
from sklearn.metrics import confusion_matrix, classification_report, f1_score

y_pred = knn_best.predict(X_test_std)
cf = confusion_matrix(y_test, y_pred)
print('Confusion matrix:\n', cf)
print('\nClassification report:\n', classification_report(y_test, y_pred, target_names=data.target_names))
print('F1 (macro):', f1_score(y_test, y_pred, average='macro'))
```

- `average='macro'`는 다중 클래스에서 클래스별 F1의 단순 평균(클래스 불균형에 민감하지 않음)
- 불균형을 고려하려면 `average='weighted'` 사용

6. **(선택) 결정 경계 시각화 — 2개 피처만 사용할 때**

```python
# 예: sepal length(0) vs sepal width(1)
import numpy as np
from matplotlib.colors import ListedColormap

X2 = X_train_std[:, :2]
X2_test = X_test_std[:, :2]
knn2 = KNeighborsClassifier(n_neighbors=k_best)
knn2.fit(X2, y_train)

# 그리드 생성
x_min, x_max = X2[:,0].min()-1, X2[:,0].max()+1
y_min, y_max = X2[:,1].min()-1, X2[:,1].max()+1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
Z = knn2.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

cmap = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap)
plt.scatter(X2[:,0], X2[:,1], c=y_train, edgecolor='k', cmap='Set1')
plt.scatter(X2_test[:,0], X2_test[:,1], c=y_test, marker='D', edgecolor='k', cmap='Set1')
plt.xlabel(data.feature_names[0])
plt.ylabel(data.feature_names[1])
plt.title('Decision boundaries (2 features)')
plt.show()
```

7. **(선택) 모델 저장**

```python
import joblib
joblib.dump({'model': knn_best, 'scaler': scaler}, 'iris_knn_joblib.pkl')
```

**팁과 주의사항**
- KNN은 데이터 수와 차원이 커질수록 계산 비용이 큽니다. 대규모 데이터엔 부적합.
- 표준화는 필수입니다(거리 기반).
- `pairplot`으로 어떤 피처 조합이 유리한지 시각적으로 판단하고, 결정 경계 시 시각화용으로 2개 피처로 제한하세요.
- 다중 클래스 F1: `macro` vs `weighted` 차이를 이해하고 사용하세요.

---

파일 저장 위치: `iris_knn_steps.md` (workspace 루트)

원하시면 이 가이드를 기반으로 실행 가능한 노트북 셀들을 추가하거나, 제가 직접 노트북에서 실행해 결과(플롯·리포트)를 생성해 드리겠습니다.