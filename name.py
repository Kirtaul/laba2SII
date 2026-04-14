import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from collections import Counter

products = []
sweetness = []
crunch = []
classes = []

with open('dataset.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        products.append(parts[0])
        sweetness.append(int(parts[1]))
        crunch.append(int(parts[2]))
        classes.append(parts[3])

X = np.column_stack((sweetness, crunch))
y = np.array(classes)

print(f"Загружено {len(X)} объектов.")
print(f"Уникальные классы: {np.unique(y)}")

class MyKNN:
    def __init__(self, k=3):
        self.k = k
    def fit(self, X, y):
        self.X = np.array(X)
        self.y = np.array(y)
    def predict(self, X):
        preds = []
        for x in X:
            # квадраты евклидовых расстояний
            dist = np.sum((self.X - x) ** 2, axis=1)
            k_idx = np.argsort(dist)[:self.k]
            k_labels = self.y[k_idx]
            preds.append(Counter(k_labels).most_common(1)[0][0])
        return np.array(preds)

#Кросс валидация
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Оценка sklearn
sk_scores = cross_val_score(KNeighborsClassifier(n_neighbors=3), X, y, cv=skf)
print(f"sklearn k-NN (k=3): {sk_scores.mean():.2f} (+/- {sk_scores.std():.2f})")

# Оценка своей реализации
my_scores = []
for train, test in skf.split(X, y):
    knn = MyKNN(k=3)
    knn.fit(X[train], y[train])
    my_scores.append(np.mean(knn.predict(X[test]) == y[test]))
print(f"MyKNN (k=3): {np.mean(my_scores):.2f} (+/- {np.std(my_scores):.2f})")

#Оптимальное k
k_range = range(1, 11)
mean_scores = [cross_val_score(KNeighborsClassifier(n_neighbors=k), X, y, cv=skf).mean() for k in k_range]
best_k = k_range[np.argmax(mean_scores)]
print(f"\nОптимальное k = {best_k} (точность = {max(mean_scores):.2f})")

plt.plot(k_range, mean_scores, 'o-')
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.title('Выбор k')
plt.grid()
plt.show()

#Обучение финальных моделей
model_sk = KNeighborsClassifier(n_neighbors=best_k).fit(X, y)
model_my = MyKNN(k=best_k)
model_my.fit(X, y)

colors = {'Фрукт':'red', 'Овощ':'green', 'Протеин':'blue', 'Зерновые':'orange'}
plt.figure(figsize=(8,6))
for cls in np.unique(y):
    mask = y == cls
    plt.scatter(X[mask,0], X[mask,1], label=cls, color=colors[cls], edgecolor='k', s=80)
plt.xlabel('Сладость')
plt.ylabel('Хруст')
plt.title('Точечный график исходных данных')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

print("ВВЕДИТЕ ДАННЫЕ НОВОГО ПРОДУКТА")
name = input("Название продукта: ").strip()
try:
    sweet = float(input("Сладость (1-10): "))
    crunch_val = float(input("Хруст (1-10): "))
    if sweet < 1 or sweet > 10 or crunch_val < 1 or crunch_val > 10:
        raise ValueError
except ValueError:
    print("Ошибка: введите числа от 1 до 10")
    exit()

# Предсказания
pred_sk = model_sk.predict([[sweet, crunch_val]])[0]
pred_my = model_my.predict([[sweet, crunch_val]])[0]
print(f"\nПредсказание sklearn (k={best_k}): {pred_sk}")
print(f"Предсказание MyKNN (k={best_k}): {pred_my}")

plt.figure(figsize=(8,6))
for cls in np.unique(y):
    mask = y == cls
    plt.scatter(X[mask,0], X[mask,1], label=cls, color=colors[cls], edgecolor='k', s=80)
# новая точка – чёрная звезда
plt.scatter(sweet, crunch_val, color='black', marker='*', s=200,
            label=f'Новый: {name} (sklearn: {pred_sk})', edgecolor='k', zorder=5)
plt.xlabel('Сладость')
plt.ylabel('Хруст')
plt.title(f'Точечный график с новым продуктом (sklearn, k={best_k})')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

plt.figure(figsize=(8,6))
for cls in np.unique(y):
    mask = y == cls
    plt.scatter(X[mask,0], X[mask,1], label=cls, color=colors[cls], edgecolor='k', s=80)
plt.scatter(sweet, crunch_val, color='black', marker='*', s=200,
            label=f'Новый: {name} (MyKNN: {pred_my})', edgecolor='k', zorder=5)
plt.xlabel('Сладость')
plt.ylabel('Хруст')
plt.title(f'Точечный график с новым продуктом (MyKNN, k={best_k})')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()
