import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
import os
import time

NORMALIZACAO = 255.0
PCA_COMPONENTES = 50
TREINO_PORCENTAGEM = 0.9
CLASSIFICADOR = 'knn'          
KNN_K = 1
KNN_METRICA = 'euclidean'
RANDOM_STATE = 42

# Nome do arquivo de saída com base nas configurações
saida_nome = f"{int(NORMALIZACAO)}_{int(TREINO_PORCENTAGEM * 100)}_{KNN_METRICA}_{CLASSIFICADOR}_{PCA_COMPONENTES}_{KNN_K}.txt"
saida_caminho = os.path.join(".", saida_nome)


inicio_tempo = time.time()


print("Carregando MNIST...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist.data, mnist.target.astype(int)

#  Normalização
X = X / NORMALIZACAO

#  PCA - Redução de dimensionalidade
print("Aplicando PCA...")
pca = PCA(n_components=PCA_COMPONENTES)
X_reduced = pca.fit_transform(X)

# Divisão dos dados
X_train, X_temp, y_train, y_temp = train_test_split(X_reduced, y, test_size=1 - TREINO_PORCENTAGEM, random_state=RANDOM_STATE)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE)

# Classificador
if CLASSIFICADOR == 'knn':
    print("\n== Classificador: kNN ==")
    clf = KNeighborsClassifier(n_neighbors=KNN_K, metric=KNN_METRICA)
else:
    print("\n== Classificador: Regressão Logística ==")
    clf = LogisticRegression(max_iter=1000)

# Treinamento
clf.fit(X_train, y_train)

#  Predições
y_val_pred = clf.predict(X_val)
y_test_pred = clf.predict(X_test)


acc_val = accuracy_score(y_val, y_val_pred)
acc_test = accuracy_score(y_test, y_test_pred)
conf_matrix = confusion_matrix(y_test, y_test_pred)


fim_tempo = time.time()
tempo_execucao = fim_tempo - inicio_tempo

with open(saida_caminho, 'w') as f:
    f.write(f"== Resultados do teste ==\n")
    f.write(f"Normalização: /{NORMALIZACAO}\n")
    f.write(f"PCA: {PCA_COMPONENTES} componentes\n")
    f.write(f"Treinamento: {int(TREINO_PORCENTAGEM * 100)}%\n")
    f.write(f"Classificador: {CLASSIFICADOR}\n")
    if CLASSIFICADOR == 'knn':
        f.write(f"k = {KNN_K}\n")
        f.write(f"Métrica: {KNN_METRICA}\n")
    f.write("\nAcurácia (Validação): {:.4f}\n".format(acc_val))
    f.write("Acurácia (Teste): {:.4f}\n".format(acc_test))
    f.write("\nMatriz de confusão (Teste):\n")
    f.write(str(conf_matrix))
    f.write("\n\nTempo de execução: {:.2f} segundos\n".format(tempo_execucao))
    f.write("\n")


print(f"\nResultados salvos em: {saida_caminho}")
