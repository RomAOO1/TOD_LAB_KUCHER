from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score
import joblib

# Завантаження датасету Iris
iris = load_iris()
X = iris.data
y = iris.target

# Поділ на тренувальну та тестову вибірки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Створення пайплайну
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Масштабування даних
    ('pca', PCA()),  # Відбір головних компонент
    ('classifier', LogisticRegression())  # Класифікація з використанням логістичної регресії
])

# Налаштування гіперпараметрів та відбір ознак за допомогою GridSearchCV
parameters = {
    'pca__n_components': [2, 3],
    'classifier__C': [0.1, 1, 10]
}

grid_search = GridSearchCV(pipeline, parameters, scoring='accuracy', cv=5)
grid_search.fit(X_train, y_train)

# Оцінка якості моделі на тестовій вибірці
y_pred = grid_search.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")

# Порівняння з базовою моделлю
baseline_model = LogisticRegression()
baseline_model.fit(X_train, y_train)
baseline_pred = baseline_model.predict(X_test)
baseline_accuracy = accuracy_score(y_test, baseline_pred)

print(f"Baseline Accuracy: {baseline_accuracy}")

# Збереження пайплайну
joblib.dump(grid_search, 'pipeline.pkl')

# Завантаження пайплайну для подальшого використання
loaded_pipeline = joblib.load('pipeline.pkl')
