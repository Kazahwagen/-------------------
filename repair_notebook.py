import json

path = 'practice_5_knn.ipynb'
with open(path, encoding='utf-8') as f:
    nb = json.load(f)

# Keep the lecture and the student's original cells; replace only the previously
# appended solution section.
cells = nb['cells'][:81]

def md(text):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': [x + '\n' for x in text.splitlines()]}

def code(text):
    return {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': [x + '\n' for x in text.splitlines()]}

cells += [
md('''## Выполнение заданий'''),
md('''В заданиях используем тот же набор признаков, который был дан в шаблоне ноутбука: `SepalLengthCm` и `SepalWidthCm`. Поэтому результаты можно сравнивать с демонстрациями из лекционной части, где границы классов показываются на плоскости.'''),
code('''# Дополнительные инструменты для выполнения заданий
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import RadiusNeighborsClassifier

# Загрузка данных по шаблону ноутбука
iris_df = pd.read_csv("./Iris.csv", index_col="Id")

X = iris_df[["SepalLengthCm", "SepalWidthCm"]].to_numpy()
encoder = LabelEncoder()
Y = encoder.fit_transform(iris_df["Species"])

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.25, random_state=42, stratify=Y
)

print("Размер обучающей выборки:", X_train.shape[0])
print("Размер тестовой выборки:", X_test.shape[0])'''),
md('''### 1. Сравнение LDA и QDA'''),
code('''def plot_classifier_boundary(model, X_train, Y_train, X_test, Y_test, title):
    xmin, xmax = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    ymin, ymax = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(xmin, xmax, 300),
        np.linspace(ymin, ymax, 300)
    )
    points = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(points).reshape(xx.shape)

    plt.figure(figsize=(7, 5))
    plt.contourf(xx, yy, Z, alpha=0.25, cmap="tab10")
    plt.scatter(X_train[:, 0], X_train[:, 1], c=Y_train, cmap="tab10", marker="o", label="обучение")
    plt.scatter(X_test[:, 0], X_test[:, 1], c=Y_test, cmap="tab10", marker="x", s=70, label="тест")
    plt.xlabel("SepalLengthCm")
    plt.ylabel("SepalWidthCm")
    plt.title(title)
    plt.legend()
    plt.show()

lda = LinearDiscriminantAnalysis()
qda = QuadraticDiscriminantAnalysis()
lda.fit(X_train, Y_train)
qda.fit(X_train, Y_train)

print("LDA, точность на тесте:", round(lda.score(X_test, Y_test), 3))
print("QDA, точность на тесте:", round(qda.score(X_test, Y_test), 3))

plot_classifier_boundary(lda, X_train, Y_train, X_test, Y_test, "LDA")
plot_classifier_boundary(qda, X_train, Y_train, X_test, Y_test, "QDA")'''),
md('''По графикам видно, как классификаторы разделяют классы на плоскости двух признаков. LDA строит линейные границы, а QDA может строить нелинейные квадратичные границы. На этом наборе признаков сравнивать методы нужно по напечатанной точности на тестовой выборке.'''),
md('''### 2. Подбор параметра k для метода kNN'''),
code('''# Для этого учебного примера оставляем исходные координаты признаков,
# чтобы границы на графике совпадали с координатами точек.
X_train_model = X_train
X_test_model = X_test

k_values = range(1, 32, 2)
accuracies = []

for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train_model, Y_train)
    accuracies.append(model.score(X_test_model, Y_test))

best_index = int(np.argmax(accuracies))
best_k = list(k_values)[best_index]
print("Лучшее значение k:", best_k)
print("Точность лучшей модели:", round(accuracies[best_index], 3))

plt.figure(figsize=(7, 4))
plt.plot(list(k_values), accuracies, marker="o")
plt.xlabel("Количество соседей k")
plt.ylabel("Точность на тестовой выборке")
plt.title("Выбор параметра k для kNN")
plt.grid(True)
plt.show()

best_knn = KNeighborsClassifier(n_neighbors=best_k)
best_knn.fit(X_train_model, Y_train)
plot_classifier_boundary(best_knn, X_train_model, Y_train, X_test_model, Y_test,
                         f"kNN, k={best_k}")'''),
md('''На графике показано, как меняется точность при разных значениях `k`. При малом `k` граница слишком чувствительна к отдельным объектам, а при большом `k` становится чрезмерно сглаженной. В качестве рабочего выбирается значение `k`, для которого точность на тестовой выборке максимальна.'''),
md('''### 3. RadiusNeighborsClassifier'''),
code('''radius_values = np.arange(0.2, 2.1, 0.1)
radius_accuracies = []

for radius in radius_values:
    model = RadiusNeighborsClassifier(
        radius=radius,
        outlier_label="most_frequent"
    )
    model.fit(X_train_model, Y_train)
    radius_accuracies.append(model.score(X_test_model, Y_test))

best_radius_index = int(np.argmax(radius_accuracies))
best_radius = radius_values[best_radius_index]
print("Лучший радиус:", round(float(best_radius), 2))
print("Точность лучшей модели:", round(radius_accuracies[best_radius_index], 3))

plt.figure(figsize=(7, 4))
plt.plot(radius_values, radius_accuracies, marker="o")
plt.xlabel("Радиус")
plt.ylabel("Точность на тестовой выборке")
plt.title("Выбор радиуса RadiusNeighborsClassifier")
plt.grid(True)
plt.show()

best_radius_model = RadiusNeighborsClassifier(
    radius=best_radius,
    outlier_label="most_frequent"
)
best_radius_model.fit(X_train_model, Y_train)
plot_classifier_boundary(
    best_radius_model, X_train_model, Y_train, X_test_model, Y_test,
    f"RadiusNeighborsClassifier, radius={best_radius:.1f}"
)'''),
md('''`RadiusNeighborsClassifier` отличается от обычного kNN тем, что заранее задаётся радиус, а число соседей определяется плотностью данных. Метод удобен, когда важно учитывать только объекты, находящиеся не дальше заданного расстояния. Его недостатки: результат сильно зависит от масштаба признаков и значения радиуса; при слишком маленьком радиусе у объекта может не оказаться соседей. В коде это обработано параметром `outlier_label="most_frequent"`.''')
]
nb['cells'] = cells
with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print('Обновлён', path, 'ячеек:', len(cells))
