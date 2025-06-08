import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar dados
df = pd.read_csv('resultados_mnist.tsv', sep="\t")

# Corrigir possíveis erros de escala
for col in ['Acc_Val', 'Acc_Test', 'Precision_Test', 'Recall_Test', 'F1_Test']:
    df[col] = df[col].apply(lambda x: x / 1000 if x > 1 else x)

# Converter colunas categóricas
df['Classificador'] = df['Classificador'].astype(str)
df['Distancia'] = df['Distancia'].astype(str)
df['Normalizacao'] = df['Normalizacao'].astype(str)

# ========= MELHORES CONFIGURAÇÕES ========= #
def imprimir_melhor_config(coluna, nome):
    max_val = df[coluna].max()
    melhores = df[df[coluna] == max_val]
    melhor_idx = melhores['Tempo_Total_s'].idxmin()
    linha = df.loc[melhor_idx]

    print(f"\n🔹 Melhor {nome}: {linha[coluna]:.4f}")
    print(f"  - Classificador: {linha['Classificador']}")
    print(f"  - PCA: {linha['PCA']}")
    print(f"  - Normalização: {linha['Normalizacao']}")
    print(f"  - k: {linha['k']} | Distância: {linha['Distancia']}")
    print(f"  - Treino%: {linha['Treino%']} | Tempo total: {linha['Tempo_Total_s']:.2f}s")


imprimir_melhor_config("Acc_Test", "Acurácia de Teste")
imprimir_melhor_config("Recall_Test", "Recall de Teste")
imprimir_melhor_config("F1_Test", "F1 Score de Teste")

# ========= GRÁFICOS ========= #
# F1 Score por PCA para cada classificador
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x="PCA", y="F1_Test", hue="Classificador")
plt.title("F1 Score por número de componentes PCA e classificador")
plt.ylabel("F1 Score (Teste)")
plt.xlabel("Componentes PCA")
plt.tight_layout()
plt.savefig("results/f1_score_por_pca_e_classificador.png")

# Precision vs Recall
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x="Recall_Test", y="Precision_Test", hue="Classificador", style="Normalizacao")
plt.title("Precisão vs Revocação (Teste)")
plt.xlabel("Recall (Teste)")
plt.ylabel("Precision (Teste)")
plt.tight_layout()
plt.savefig("results/precision_vs_recall.png")

# Tempo Total vs Acurácia de Teste
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x="Tempo_Total_s", y="Acc_Test", hue="Classificador", style="Distancia")
plt.title("Trade-off: Tempo Total vs Acurácia de Teste")
plt.xlabel("Tempo Total (s)")
plt.ylabel("Acurácia no Teste")
plt.tight_layout()
plt.savefig("results/tempo_total_vs_acc_test.png")

# Overfitting (Acc_Val - Acc_Test)
df["Delta_Acc"] = df["Acc_Val"] - df["Acc_Test"]
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x="Classificador", y="Delta_Acc", hue="PCA")
plt.title("Diferença entre Acurácia de Validação e Teste (Overfitting)")
plt.ylabel("Acc_Val - Acc_Test")
plt.tight_layout()
plt.savefig("results/overfitting_acc_val_vs_test.png")

# Heatmap F1 médio por k e distância (kNN apenas)
df_knn = df[df["Classificador"] == "knn"]
pivot_f1 = df_knn.pivot_table(index="k", columns="Distancia", values="F1_Test", aggfunc="mean")
plt.figure(figsize=(8, 6))
sns.heatmap(pivot_f1, annot=True, fmt=".3f", cmap="YlGnBu")
plt.title("F1 médio por k e distância (kNN)")
plt.tight_layout()
plt.savefig("results/f1_por_k_e_distancia_knn.png")

print("\n✅ Análise concluída e gráficos salvos com sucesso!")
