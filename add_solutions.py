import json

notebook_path = 'practice_5_knn.ipynb'
output_path = 'practice_5_knn_completed.ipynb'

with open(notebook_path, encoding='utf-8') as f:
    nb = json.load(f)

def markdown(text):
    return {
        'cell_type': 'markdown',
        'metadata': {},
        'source': [line + '\n' for line in text.splitlines()]
    }

def code(text):
    return {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': [line + '\n' for line in text.splitlines()]
    }

solutions = [
markdown('''# Решения домашнего задания

Для всех экспериментов ниже используется датасет Iris. Целевая переменная кодируется числами классов, а разбиение на обучающую и тестовую выборки стратифицировано. Это сохраняет одинаковую долю каждого класса в обеих выборках.'''),
code('''from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.neighbors import RadiusNeighborsClassifier

iris_df = pd.read_csv("Iris.csv", index_col="Id")
feature_names = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
X = iris_df[feature_names]
y = LabelEncoder().fit_transform(iris_df["Species"])
class_names = np.sort(iris_df["Species"].unique())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
print(f"Размер обучающей выборки: {X_train.shape[0]}")
print(f"Размер тестовой выборки: {X_test.shape[0]}")'''),
markdown('''## 1. Сравнение LDA и QDA

LDA предполагает общую ковариационную матрицу для всех классов, поэтому границы решений у него линейные. QDA строит отдельную ковариационную матрицу для каждого класса и способен формировать квадратичные границы. Оценим оба метода на отложенной тестовой выборке.'''),
code('''models = {
    "LDA": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearDiscriminantAnalysis())
    ]),
    "QDA": Pipeline([
        ("scaler", StandardScaler()),
        ("model", QuadraticDiscriminantAnalysis())
    ])
}

comparison = []
predictions = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    predictions[name] = y_pred
    comparison.append({
        "Метод": name,
        "Accuracy (train)": model.score(X_train, y_train),
        "Accuracy (test)": accuracy_score(y_test, y_pred)
    })

comparison_df = pd.DataFrame(comparison).set_index("Метод")
display(comparison_df.round(3))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, (name, y_pred) in zip(axes, predictions.items()):
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=class_names, cmap="Blues", ax=ax, colorbar=False
    )
    ax.set_title(name)
plt.tight_layout()'''),
markdown('''**Вывод.** На данном разбиении сравниваются значения `Accuracy (test)` в таблице выше. У Iris классы хорошо разделимы, поэтому оба дискриминантных метода обычно дают высокую точность. Если результат QDA на тесте ниже LDA, это иллюстрирует более высокую гибкость QDA: отдельные ковариационные матрицы могут переобучаться на небольшой выборке. LDA является более устойчивым вариантом при близких ковариациях классов; QDA стоит выбирать, когда есть достаточно данных и форма классов заметно различается.'''),
markdown('''## 2. Подбор параметров kNN

Масштабы признаков влияют на расстояния, поэтому `StandardScaler` включён в конвейер. Лучшие параметры выбираются только по обучающей части данных с помощью стратифицированной 5-fold кросс-валидации. Затем качество оценивается один раз на тестовой выборке.'''),
code('''knn_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier())
])

param_grid = {
    "knn__n_neighbors": list(range(1, 32, 2)),
    "knn__weights": ["uniform", "distance"],
    "knn__metric": ["euclidean", "manhattan"]
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
search = GridSearchCV(knn_pipeline, param_grid, cv=cv, scoring="accuracy", n_jobs=-1)
search.fit(X_train, y_train)

best_knn = search.best_estimator_
y_pred_knn = best_knn.predict(X_test)
print("Лучшие параметры:", search.best_params_)
print(f"Лучшая accuracy на CV: {search.best_score_:.3f}")
print(f"Accuracy на тестовой выборке: {accuracy_score(y_test, y_pred_knn):.3f}")
print("\\nОтчёт по классам:")
print(classification_report(y_test, y_pred_knn, target_names=class_names, zero_division=0))

cv_results = pd.DataFrame(search.cv_results_)
plot_data = cv_results[
    (cv_results["param_knn__weights"] == search.best_params_["knn__weights"]) &
    (cv_results["param_knn__metric"] == search.best_params_["knn__metric"])
].sort_values("param_knn__n_neighbors")
plt.figure(figsize=(8, 4))
plt.plot(plot_data["param_knn__n_neighbors"], plot_data["mean_test_score"], marker="o")
plt.axvline(search.best_params_["knn__n_neighbors"], color="red", linestyle="--", label="лучшее k")
plt.xlabel("Число соседей k")
plt.ylabel("Средняя accuracy на CV")
plt.title("Подбор k для лучших weights и metric")
plt.legend()
plt.grid(alpha=0.3)
plt.show()'''),
markdown('''**Вывод.** Оптимальная комбинация параметров и её кросс-валидационная точность выводятся в ячейке выше. Небольшое `k` даёт очень гибкую границу и чувствительно к отдельным наблюдениям, а чрезмерно большое `k` сглаживает границу и может ухудшить качество. Поиск по CV позволяет выбрать баланс без использования тестовой выборки при настройке.'''),
markdown('''## 3. RadiusNeighborsClassifier

`RadiusNeighborsClassifier` ищет **всех** объектов в заданном радиусе, а не фиксированное число ближайших соседей. Класс определяется голосованием этих соседей. При отсутствии соседей метод обычно вызывает ошибку; здесь указан `outlier_label="most_frequent"`, поэтому такой объект получает наиболее частый класс обучающей выборки.'''),
code('''radius_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("radius", RadiusNeighborsClassifier(outlier_label="most_frequent"))
])

radius_grid = {"radius__radius": np.arange(0.3, 2.05, 0.1)}
radius_search = GridSearchCV(radius_pipeline, radius_grid, cv=cv, scoring="accuracy", n_jobs=-1)
radius_search.fit(X_train, y_train)

best_radius = radius_search.best_estimator_
y_pred_radius = best_radius.predict(X_test)
print("Лучший радиус:", radius_search.best_params_["radius__radius"])
print(f"Лучшая accuracy на CV: {radius_search.best_score_:.3f}")
print(f"Accuracy на тестовой выборке: {accuracy_score(y_test, y_pred_radius):.3f}")

radius_results = pd.DataFrame(radius_search.cv_results_).sort_values("param_radius__radius")
plt.figure(figsize=(8, 4))
plt.plot(radius_results["param_radius__radius"], radius_results["mean_test_score"], marker="o")
plt.axvline(radius_search.best_params_["radius__radius"], color="red", linestyle="--", label="лучший радиус")
plt.xlabel("Радиус")
plt.ylabel("Средняя accuracy на CV")
plt.title("Подбор радиуса RadiusNeighborsClassifier")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred_radius, display_labels=class_names, cmap="Greens", colorbar=False
)
plt.title("RadiusNeighborsClassifier")
plt.show()'''),
markdown('''### Достоинства и недостатки RadiusNeighborsClassifier

**Достоинства:**
- число используемых соседей адаптируется к плотности данных: в плотной области их больше, в разреженной — меньше;
- радиус имеет понятную интерпретацию как допустимая близость объектов;
- метод может выявлять необычные или удалённые объекты: для них соседей в радиусе не будет.

**Недостатки:**
- параметр радиуса очень чувствителен к масштабу признаков, поэтому стандартизация необходима;
- при малом радиусе возможны объекты без соседей; требуется обработка через `outlier_label` или иной сценарий;
- в плотных областях число соседей может быть большим, что увеличивает вычисления и размывает локальное решение;
- как и kNN, метод хранит обучающие данные, чувствителен к нерелевантным признакам и дорог при предсказании на больших выборках.

В этом задании радиус, как и `k`, подобран кросс-валидацией; итоговая точность показана в выводе ячейки выше.''')
]

nb['cells'].extend(solutions)
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f'Создан файл: {output_path}')
