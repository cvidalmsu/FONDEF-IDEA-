import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from collections import deque
import random
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef, balanced_accuracy_score

# ✅ Preprocesamiento de datos con manejo de NaN
def preprocess_data(data):
    data = data.drop(columns=["Fecha ultimo Poligono"], errors='ignore')
    data = data.fillna(0)  # ✅ Rellenar valores faltantes con 0
    X = data.drop(columns=["Fire_Probability"]).values
    y = data["Fire_Probability"].values
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1  # ✅ Evitar división por cero
    X = (X - mean) / std

    # ✅ Verificación para detener si hay NaN
    if np.isnan(X).any() or np.isnan(y).any():
        raise ValueError("Error: Se encontraron valores NaN después del preprocesamiento.")
    
    return X, y

# ✅ Cargar y validar datos
training_data = pd.read_csv(r"C:\Users\cvshi\Desktop\FONDEF TI\PYTHON\Datos\extended_training_data.csv")
validation_data = pd.read_csv(r"C:\Users\cvshi\Desktop\FONDEF TI\PYTHON\Datos\extended_validation_data.csv")

X_train, y_train = preprocess_data(training_data)
X_val, y_val = preprocess_data(validation_data)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

batch_size = 64
train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=batch_size)

# ✅ Definición del modelo Actor-Critic
class ActorCriticNetwork(nn.Module):
    def __init__(self, input_dim):
        super(ActorCriticNetwork, self).__init__()
        self.common = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.actor = nn.Sequential(
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        self.critic = nn.Linear(64, 1)  # Sin Sigmoid, lo aplicaremos después

    def forward(self, x):
        x = self.common(x)
        action_prob = self.actor(x)
        value = self.critic(x)
        return action_prob, value

input_dim = X_train.shape[1]
actor_critic_model = ActorCriticNetwork(input_dim)
optimizer = optim.Adam(actor_critic_model.parameters(), lr=0.001)
loss_fn = nn.BCELoss()

# ✅ Implementación del Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state):
        self.buffer.append((state, action, reward, next_state))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def size(self):
        return len(self.buffer)

# ✅ Entrenamiento del modelo con validación anti-NaN
def train_actor_critic(epochs=20):
    replay_buffer = ReplayBuffer(10000)
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            action_probs, values = actor_critic_model(X_batch)
            values = torch.sigmoid(values)

            # ✅ Aplicar validación contra NaN y valores fuera de rango
            rewards = y_batch.float().view(-1, 1)
            rewards = torch.clamp(rewards, min=0.0, max=1.0)

            if torch.isnan(values).any() or torch.isnan(rewards).any():
                raise ValueError("Error: Se encontraron valores NaN durante el entrenamiento.")

            # ✅ Cálculo de la pérdida
            critic_loss = loss_fn(values, rewards)
            advantage = rewards - values.detach()
            actor_loss = -(action_probs * advantage).mean()
            total_loss = actor_loss + critic_loss

            # Actualización de los pesos
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            replay_buffer.add(X_batch, action_probs.detach(), rewards, values.detach())

            epoch_loss += total_loss.item()
        print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")

# ✅ Evaluación con métricas completas
def evaluate_model_with_metrics():
    actor_critic_model.eval()
    predictions, y_true, y_pred, y_prob = [], [], [], []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            action_probs, _ = actor_critic_model(X_batch)
            predicted_fire = (action_probs >= 0.5).int()
            y_true.extend(y_batch.flatten().tolist())
            y_pred.extend(predicted_fire.flatten().tolist())
            y_prob.extend(action_probs.flatten().tolist())

    # ✅ Cálculo de métricas
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc_roc = roc_auc_score(y_true, y_prob)
    mcc = matthews_corrcoef(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    
    # ✅ Imprimir resultados
    print(f"\n--- Métricas de Evaluación ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC-ROC: {auc_roc:.4f}")
    print(f"MCC: {mcc:.4f}")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")

# ✅ Ejecutar entrenamiento y evaluación final
train_actor_critic(epochs=20)
evaluate_model_with_metrics()

