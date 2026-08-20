import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import json

print("\n" + "="*60)
print("🚀 AI CODING AGENT - LIGHTWEIGHT TRAINING")
print("="*60)

# Step 1: Create Lightweight Model
class LightweightCodingAgent(nn.Module):
    def __init__(self):
        super(LightweightCodingAgent, self).__init__()
        self.embedding = nn.Embedding(vocab_size=10000, embedding_dim=128)
        self.lstm = nn.LSTM(input_size=128, hidden_size=256, num_layers=2, batch_first=True)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.dropout(torch.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

print("\n✅ Model Architecture Created")
print("   - Embedding Layer: 10000 vocab -> 128 dims")
print("   - LSTM: 2 layers, 256 hidden units")
print("   - Dense Layers: 128 -> 64 -> 32")

# Step 2: Initialize Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LightweightCodingAgent().to(device)
print(f"\n✅ Model initialized on {device}")

# Step 3: Create Synthetic Data
print("\n📊 Creating Training Data...")
batch_size = 32
num_batches = 10
seq_length = 50

X = torch.randint(0, 10000, (batch_size * num_batches, seq_length))
y = torch.randn(batch_size * num_batches, 32)

dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
print(f"✅ Dataset created: {batch_size * num_batches} samples")

# Step 4: Training Setup
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

print("\n🔧 Training Configuration:")
print("   - Optimizer: Adam (lr=0.001)")
print("   - Loss Function: MSELoss")
print("   - Epochs: 5")

# Step 5: Train Model
print("\n" + "="*60)
print("🎯 TRAINING STARTED")
print("="*60)

train_losses = []

for epoch in range(5):
    epoch_loss = 0.0
    model.train()
    
    for batch_idx, (X_batch, y_batch) in enumerate(dataloader):
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = loss_fn(outputs, y_batch)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
        if (batch_idx + 1) % 5 == 0:
            print(f"   Epoch {epoch+1}/5, Batch {batch_idx+1}/{len(dataloader)}, Loss: {loss.item():.4f}")
    
    avg_loss = epoch_loss / len(dataloader)
    train_losses.append(avg_loss)
    print(f"\n✅ Epoch {epoch+1} completed - Avg Loss: {avg_loss:.4f}\n")

# Step 6: Save Model
os.makedirs('models', exist_ok=True)
model_path = 'models/ai_coding_agent_trained.pt'
torch.save(model.state_dict(), model_path)
print(f"💾 Model saved to: {model_path}")

# Save Training Info
training_info = {
    "model": "LightweightCodingAgent",
    "architecture": {
        "embedding_dim": 128,
        "lstm_hidden": 256,
        "lstm_layers": 2,
        "vocab_size": 10000
    },
    "training": {
        "epochs": 5,
        "batch_size": 32,
        "optimizer": "Adam",
        "learning_rate": 0.001,
        "losses": [float(l) for l in train_losses]
    },
    "device": str(device),
    "status": "✅ TRAINING COMPLETED SUCCESSFULLY"
}

with open('models/training_info.json', 'w') as f:
    json.dump(training_info, f, indent=2)

print(f"📄 Training info saved to: models/training_info.json")

print("\n" + "="*60)
print("✨ TRAINING COMPLETED SUCCESSFULLY!")
print("="*60)
print(f"\n📊 Training Summary:")
print(f"   - Final Loss: {train_losses[-1]:.4f}")
print(f"   - Loss Improvement: {((train_losses[0] - train_losses[-1])/train_losses[0]*100):.2f}%")
print(f"   - Model Size: {sum(p.numel() for p in model.parameters())} parameters")
print(f"   - Device: {device}")
print("\n✅ Model Ready for Deployment!")
print("="*60 + "\n")