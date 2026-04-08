import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# 1단계. 패키지 설정
# 2단계. 데이터 준비
data = load_iris(as_frame=True)
df = data.frame.copy()
df["species"] = df["target"].map(dict(enumerate(data.target_names)))

X = data.data
y = data.target

print("데이터 크기:", X.shape)
print("클래스 이름:", data.target_names)
print("클래스 분포:\n", df["species"].value_counts())
print("\n기초 통계량:\n", df[data.feature_names].describe())
print("\n품종별 평균:\n", df.groupby("species")[data.feature_names].mean().round(2))

# 3단계. 탐색적 데이터 분석
sns.set_theme(style="whitegrid")
pair = sns.pairplot(df, vars=data.feature_names, hue="species", diag_kind="hist")
pair.fig.suptitle("Iris pairplot", y=1.02)
plt.show()

# 4단계. 피처 스케일링 전 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

print("\n학습 데이터 크기:", X_train.shape)
print("테스트 데이터 크기:", X_test.shape)
print("\n표준화된 학습 데이터 일부:\n", X_train_std[:5])

# 5단계. 모형화 및 학습
neighbors = range(1, 16)
train_acc = []
test_acc = []

for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_std, y_train)
    train_acc.append(knn.score(X_train_std, y_train))
    test_acc.append(knn.score(X_test_std, y_test))

best_k = list(neighbors)[int(np.argmax(test_acc))]
print("\n각 k별 테스트 정확도:")
for k, score in zip(neighbors, test_acc):
    print(f"k={k:2d} -> {score:.4f}")
print("\n선택한 최적 k:", best_k)

plt.figure(figsize=(8, 5))
plt.plot(list(neighbors), train_acc, marker="o", label="Train accuracy")
plt.plot(list(neighbors), test_acc, marker="s", label="Test accuracy")
plt.xlabel("k")
plt.ylabel("Accuracy")
plt.xticks(list(neighbors))
plt.legend()
plt.tight_layout()
plt.show()

# 최종 모델 학습
knn_best = KNeighborsClassifier(n_neighbors=best_k)
knn_best.fit(X_train_std, y_train)

# 6단계. 예측 및 평가
y_pred = knn_best.predict(X_test_std)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\n최종 정확도:", round(acc, 4))
print("\n혼동행렬:\n", cm)
print("\n분류 보고서:\n", classification_report(y_test, y_pred, target_names=data.target_names))

# 새 샘플 예측 예시
new_sample = pd.DataFrame([[5.7, 2.9, 4.2, 1.3]], columns=data.feature_names)
new_sample_std = scaler.transform(new_sample)
new_pred = knn_best.predict(new_sample_std)
new_prob = knn_best.predict_proba(new_sample_std)

print("새 샘플:", new_sample.values.tolist()[0])
print("예측 품종:", data.target_names[new_pred[0]])
print("클래스별 확률:", new_prob)
