import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import json

class AICodingAgent(nn.Module):
    def __init__(self, config_path='config.json'):
        super(AICodingAgent, self).__init__()
        
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Load pretrained BERT model
        model_name = self.config['model']['name']
        self.bert = BertModel.from_pretrained(model_name)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        
        # Classification head
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, 256)
        self.output_layer = nn.Linear(256, 128)
        
        print("✅ Model initialized successfully")
    
    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        pooled_output = outputs.pooler_output
        dropped = self.dropout(pooled_output)
        logits = self.classifier(dropped)
        logits = torch.relu(logits)
        final_output = self.output_layer(logits)
        
        return final_output
    
    def save_model(self, path):
        torch.save(self.state_dict(), path)
        print(f"✅ Model saved to {path}")
    
    def load_model(self, path):
        self.load_state_dict(torch.load(path))
        print(f"✅ Model loaded from {path}")

if __name__ == "__main__":
    model = AICodingAgent()
    print("Model ready for training!")