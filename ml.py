# =======================================
# Prática de IA - Machine Learning
# Nome: Vinícius Teixeira de Moraes Luiz
# 5º Semestre de ADS
# =======================================

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

# 1. CARREGAR DADOS
dados = load_iris()
X = dados.data
y = dados.target

# 2. DIVIDIR DADOS
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 3. MODELO 1 (ÁRVORE DE DECISÃO)
modelo_arvore = DecisionTreeClassifier()
modelo_arvore.fit(X_treino, y_treino)
pred_arvore = modelo_arvore.predict(X_teste)
acc_arvore = accuracy_score(y_teste, pred_arvore)

# 3. MODELO 2 (KNN - K-NEAREST NEIGHBORS)
modelo_knn = KNeighborsClassifier()
modelo_knn.fit(X_treino, y_treino)
pred_knn = modelo_knn.predict(X_teste)
acc_knn = accuracy_score(y_teste, pred_knn)

# 4. RESULTADOS NO TERMINAL
print(f"Acurácia Árvore de Decisão: {acc_arvore:.4f}")
print(f"Acurácia KNN: {acc_knn:.4f}")

# 5. MATRIZ DE CONFUSÃO (usando o modelo de Árvore como exemplo)
matriz = confusion_matrix(y_teste, pred_arvore)
print("\nMatriz de Confusão (Árvore):\n", matriz)

# 6. GRÁFICOS DE COMPARAÇÃO
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Gráfico 1: Comparação de Acurácia
modelos = ["Árvore", "KNN"]
valores = [acc_arvore, acc_knn]
ax[0].bar(modelos, valores, color=['skyblue', 'lightgreen'])
ax[0].set_title("Comparação de Modelos (Acurácia)")
ax[0].set_ylabel("Acurácia")
ax[0].set_ylim(0, 1.1)
for i, v in enumerate(valores):
    ax[0].text(i, v + 0.02, f"{v:.2f}", ha='center')

# Gráfico 2: Matriz de Confusão
img = ax[1].imshow(matriz, cmap='Blues')
ax[1].set_title("Matriz de Confusão (Árvore)")
classes = ["Setosa", "Versicolor", "Virginica"]
ax[1].set_xticks(range(len(classes)))
ax[1].set_yticks(range(len(classes)))
ax[1].set_xticklabels(classes)
ax[1].set_yticklabels(classes)
ax[1].set_xlabel("Previsto")
ax[1].set_ylabel("Real")

for i in range(len(classes)):
    for j in range(len(classes)):
        ax[1].text(j, i, matriz[i, j], ha="center", va="center", 
                  color="white" if matriz[i, j] > (matriz.max()/2) else "black")

fig.colorbar(img, ax=ax[1])
plt.tight_layout()

plt.savefig('resultado_ml.png')
print("\nGráfico salvo como 'resultado_ml.png'")