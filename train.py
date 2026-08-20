import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from model import AICodingAgent
import json
import numpy as np
from tqdm import tqdm
import os

def load_config(config_path='config.json'):
    with open(config_path, 'r') as f:
        return json.load(f)

def create_dummy_dataset(config):
    """Create dummy dataset for training"""
    print("📊 Creating training dataset...")
    
    size = config['data']['dataset_size']
    max_seq_length = config['training']['max_seq_length']
    batch_size = config['training']['batch_size']
    
    # Create dummy data
    input_ids = torch.randint(0, 30522, (size, max_seq_length))
    attention_mask = torch.ones(size, max_seq_length)
    token_type_ids = torch.zeros(size, max_seq_length, dtype=torch.long)
    labels = torch.randint(0, 128, (size,))
    
    dataset = TensorDataset(input_ids, attention_mask, token_type_ids, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print(f"✅ Dataset created: {size} samples")
    return dataloader

def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    
    progress_bar = tqdm(dataloader, desc="Training")
    for batch_idx, (input_ids, attention_mask, token_type_ids, labels) in enumerate(progress_bar):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        token_type_ids = token_type_ids.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(input_ids, attention_mask, token_type_ids)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss

def main():
    print("🚀 Starting AI Coding Agent Training...")
    print("=" * 50)
    
    # Load configuration
    config = load_config()
    print(f"✅ Configuration loaded")
    print(f"   Epochs: {config['training']['epochs']}")
    print(f"   Batch Size: {config['training']['batch_size']}")
    print(f"   Learning Rate: {config['training']['learning_rate']}")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✅ Device: {device}")
    
    # Create model
    model = AICodingAgent(config_path='config.json')
    model.to(device)
    
    # Create dataset
    dataloader = create_dummy_dataset(config)
    
    # Optimizer and Loss
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    criterion = nn.MSELoss()
    
    # Create output directories
    os.makedirs(config['output']['model_dir'], exist_ok=True)
    os.makedirs(config['output']['checkpoint_dir'], exist_ok=True)
    
    # Training loop
    print("\n" + "=" * 50)
    print("🎯 Starting Training Loop...")
    print("=" * 50)
    
    for epoch in range(config['training']['epochs']):
        print(f"\n📍 Epoch {epoch+1}/{config['training']['epochs']}")
        avg_loss = train_epoch(model, dataloader, optimizer, criterion, device)
        print(f"   Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        checkpoint_path = os.path.join(
            config['output']['checkpoint_dir'],
            f"checkpoint_epoch_{epoch+1}.pt"
        )
        torch.save(model.state_dict(), checkpoint_path)
        print(f"   ✅ Checkpoint saved: {checkpoint_path}")
    
    # Save final model
    model_path = os.path.join(config['output']['model_dir'], 'ai_coding_agent_final.pt')
    model.save_model(model_path)
    
    print("\n" + "=" * 50)
    print("✨ Training completed successfully!")
    print(f"📁 Model saved to: {model_path}")
    print("=" * 50)

if __name__ == "__main__":
    main()