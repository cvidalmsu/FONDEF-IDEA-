
import matplotlib.pyplot as plt
import pandas as pd

# Cargar datos de los resultados (simulados para este ejemplo)
epochs = list(range(1, 21))
losses = [8.1122, 4.4137, 1.9805, 0.4863, 0.1444, 0.0720, 0.0468, 0.0342, 0.0262, 0.0205, 0.0168, 0.0137, 0.0116, 0.0099, 0.0085, 0.0074, 0.0065, 0.0058, 0.0052, 0.0047]

# Métricas finales
metrics = {
    "Accuracy": 0.9930,
    "Precision": 1.0000,
    "Recall": 0.9895,
    "F1 Score": 0.9947,
    "AUC-ROC": 1.0000,
    "MCC": 0.9844,
    "Balanced Accuracy": 0.9948
}

# Gráfico de la pérdida a lo largo de las épocas
plt.figure(figsize=(10, 6))
plt.plot(epochs, losses, marker='o')
plt.title('Loss during Training')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.savefig('./loss_plot.png')
plt.show()

# Gráfico de barras de métricas
plt.figure(figsize=(10, 6))
plt.bar(metrics.keys(), metrics.values(), color='skyblue')
plt.title('Final Evaluation Metrics')
plt.ylabel('Score')
plt.ylim(0.95, 1.05)  # Para destacar que los valores son altos
plt.grid(axis='y')
plt.savefig('./metrics_plot.png')
plt.show()

print("Los gráficos han sido generados: 'loss_plot.png' y 'metrics_plot.png'")
