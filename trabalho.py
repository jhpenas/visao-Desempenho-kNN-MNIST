import numpy as np
import os
import time
import itertools
import csv
from datetime import datetime
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, accuracy_score, classification_report,
    precision_score, recall_score, f1_score
)

print("Carregando MNIST...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X_original, y = mnist.data, mnist.target.astype(int)

RANDOM_STATE = 42
csv_path = "resultados_mnist.tsv"  # CSV único acumulativo


def salvar_resultados_tsv(
    timestamp, normalizacao, pca_componentes, treino_porcentagem, classificador, knn_k, distancia_metrica,
    acc_val, acc_test, prec_test, rec_test, f1_test,
    tempo_treino, tempo_pred, tempo_total
):
    # Escreve em modo append e cria cabeçalho só se arquivo não existir
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        if not file_exists:
            writer.writerow([
                'Timestamp', 'Normalizacao', 'PCA', 'Treino%', 'Classificador', 'k', 'Distancia',
                'Acc_Val', 'Acc_Test', 'Precision_Test', 'Recall_Test', 'F1_Test',
                'Tempo_Treino_s', 'Tempo_Pred_s', 'Tempo_Total_s'
            ])
        writer.writerow([
            timestamp, normalizacao, pca_componentes, treino_porcentagem, classificador,
            knn_k if knn_k is not None else '',
            distancia_metrica if distancia_metrica is not None else '',
            f"{acc_val:.3f}", f"{acc_test:.3f}", f"{prec_test:.3f}", f"{rec_test:.3f}", f"{f1_test:.3f}",
            f"{tempo_treino:.6f}", f"{tempo_pred:.6f}", f"{tempo_total:.6f}"
        ])

    print(f"\nResultado da rodada salvo no arquivo {csv_path}")


def normalizar(X, metodo):
    if metodo == 'padrao':
        std = X.std(axis=0)
        std[std == 0] = 1  # evita zero
        return (X - X.mean(axis=0)) / std
    elif metodo == '255':
        return X / 255.0
    elif metodo == '[-1,1]':
        return (X - 127.5) / 127.5
    else:  # 'sem'
        return X


# Cache para PCA (normalizacao, pca_componentes) -> X_reduced
pca_cache = {}

# Parâmetros
lst_normalizacao = ['padrao', '255', '[-1,1]', 'sem']
lst_pca = [10, 50, 100, 150]
lst_treino_size = [0.6, 0.9]
lst_classificador = ['knn', 'log']
lst_knn_k = [1, 3, 5, 7]
lst_distancia = ['euclidean', 'manhattan', 'chebyshev']
lst_gridsearch = [False]  # gridsearch removido

combinacoes = list(itertools.product(
    lst_normalizacao,
    lst_pca,
    lst_treino_size,
    lst_classificador,
    lst_knn_k,
    lst_distancia,
    lst_gridsearch
))

print(f"Total de combinações a executar: {len(combinacoes)}")


def rodar_modelo_com_cache(normalizacao, pca_componentes, treino_porcentagem,
                           classificador, knn_k, distancia_metrica, X_reduced):
    inicio_tempo_total = time.time()

    # Divisão dos dados
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_reduced, y, test_size=1 - treino_porcentagem, random_state=RANDOM_STATE)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE)

    if classificador == 'knn':
        clf = KNeighborsClassifier(n_neighbors=knn_k, metric=distancia_metrica, n_jobs=-1)
        inicio_treino = time.time()
        clf.fit(X_train, y_train)
        tempo_treino = time.time() - inicio_treino
    else:
        clf = LogisticRegression(
            max_iter=300,
            tol=1e-3,
            solver='saga',
            multi_class='multinomial',
            n_jobs=-1,
            random_state=RANDOM_STATE
        )
        inicio_treino = time.time()
        clf.fit(X_train, y_train)
        tempo_treino = time.time() - inicio_treino

    inicio_pred = time.time()
    y_val_pred = clf.predict(X_val)
    y_test_pred = clf.predict(X_test)
    tempo_pred = time.time() - inicio_pred

    acc_val = accuracy_score(y_val, y_val_pred)
    acc_test = accuracy_score(y_test, y_test_pred)
    prec_test = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
    rec_test = recall_score(y_test, y_test_pred, average='weighted')
    f1_test = f1_score(y_test, y_test_pred, average='weighted')

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tempo_total = time.time() - inicio_tempo_total

    salvar_resultados_tsv(
        timestamp, normalizacao, pca_componentes, treino_porcentagem, classificador, knn_k, distancia_metrica,
        acc_val, acc_test, prec_test, rec_test, f1_test,
        tempo_treino, tempo_pred, tempo_total
    )


for i, (norm, pca_n, treino_pct, clf, k, dist, gs) in enumerate(combinacoes):
    print(f"\nExecutando combinação {i + 1} de {len(combinacoes)}...")

    if clf == 'log':
        k = None
        dist = None

    key_cache = (norm, pca_n)
    if key_cache not in pca_cache:
        X_norm = normalizar(X_original, norm)
        pca = PCA(n_components=pca_n, random_state=RANDOM_STATE)
        X_reduced = pca.fit_transform(X_norm)
        pca_cache[key_cache] = X_reduced
    else:
        X_reduced = pca_cache[key_cache]

    try:
        rodar_modelo_com_cache(norm, pca_n, treino_pct, clf, k, dist, X_reduced)
    except Exception as e:
        print(f"Erro na combinação {i + 1}: {e}")
        continue
