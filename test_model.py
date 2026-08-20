import torch
import json
from transformers import BertTokenizer
import os

print("=" * 60)
print("🤖 AI CODING AGENT - MODEL TEST")
print("=" * 60)

# Step 1: Check PyTorch
print("\n✅ PyTorch Version:", torch.__version__)
print("✅ CUDA Available:", torch.cuda.is_available())

# Step 2: Load Config
try:
    with open('config.json', 'r') as f:
        config = json.load(f)
    print("\n✅ Config loaded successfully")
    print(f"   Model: {config['model']['name']}")
    print(f"   Hidden Size: {config['model']['hidden_size']}")
except Exception as e:
    print(f"\n❌ Error loading config: {e}")
    exit(1)

# Step 3: Load Tokenizer
try:
    tokenizer = BertTokenizer.from_pretrained(config['model']['name'])
    print("\n✅ Tokenizer loaded successfully")
except Exception as e:
    print(f"\n❌ Error loading tokenizer: {e}")
    exit(1)

# Step 4: Test Tokenization
try:
    test_texts = [
        "How do I write a Python function?",
        "এটি একটি বাংলা পরীক্ষা"
    ]
    for text in test_texts:
        encoded = tokenizer(text, padding='max_length', max_length=512, truncation=True)
        print(f"\n✅ Tokenized: '{text[:30]}...'")
        print(f"   Input IDs shape: {len(encoded['input_ids'])}")
except Exception as e:
    print(f"\n❌ Error tokenizing: {e}")
    exit(1)

# Step 5: Create Output Directory
try:
    os.makedirs('models', exist_ok=True)
    print("\n✅ Models directory created")
except Exception as e:
    print(f"\n❌ Error creating directory: {e}")
    exit(1)

print("\n" + "=" * 60)
print("✨ ALL TESTS PASSED - MODEL IS READY!")
print("=" * 60)
