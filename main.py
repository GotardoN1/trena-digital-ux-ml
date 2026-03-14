import sklearn
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split

# 1. Carregar o dataset MNIST (Dígitos manuscritos)
digits = datasets.load_digits()

# 2. Preparar os dados (Flatten nas imagens)
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

# 3. Criar o classificador SVM (O coração do seu projeto)
clf = svm.SVC(gamma=0.001)

# 4. Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    data, digits.target, test_size=0.5, shuffle=False
)

# 5. Treinar o modelo
clf.fit(X_train, y_train)

# 6. Prever e mostrar resultado
predicted = clf.predict(X_test)
print(f"Relatório de Classificação:\n{metrics.classification_report(y_test, predicted)}")
