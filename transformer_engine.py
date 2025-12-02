# transformer_engine.py
import torch
from torch import nn
import torch.optim as optim
from keras._tf_keras.keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
import numpy as np

# --- INIT PARAMETERS ---
VOCAB_SIZE = 1000
MAX_LEN = 20
D_MODEL = 64
NUM_HEADS = 4
DROP_PROB = 0.1
FFN_HIDDEN = 128
NUM_CLASSES = 3  # IVC, Maurya, British
NUM_LAYERS = 3

class IndianHistoryTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        
        # --- EMBEDDING & POSITIONAL ENCODING ---
        self.emb = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.pos = nn.Embedding(MAX_LEN, D_MODEL)
        
        # --- LAYERS (Lists for Stacking) ---
        self.mha_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=D_MODEL, num_heads=NUM_HEADS, batch_first=True)
            for _ in range(NUM_LAYERS)
        ])
        
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(D_MODEL, FFN_HIDDEN),
                nn.ReLU(),
                nn.Dropout(DROP_PROB),
                nn.Linear(FFN_HIDDEN, D_MODEL)
            ) for _ in range(NUM_LAYERS)
        ])
        
        self.ln1_layers = nn.ModuleList([nn.LayerNorm(D_MODEL) for _ in range(NUM_LAYERS)])
        self.ln2_layers = nn.ModuleList([nn.LayerNorm(D_MODEL) for _ in range(NUM_LAYERS)])
        
        self.drop1 = nn.Dropout(DROP_PROB)
        self.drop2 = nn.Dropout(DROP_PROB)
        
        # --- CLASSIFIER HEAD ---
        self.classifier = nn.Linear(D_MODEL, NUM_CLASSES)

    def forward(self, x):
        # Position IDs
        B, T = x.shape
        position_ids = torch.arange(T, device=x.device).unsqueeze(0).repeat(B, 1)
        
        # Embed Input + Position
        x = self.emb(x) + self.pos(position_ids)
        
        # --- ENCODER STACK (Manual Loop) ---
        for i in range(NUM_LAYERS):
            # 1. Multi-Head Attention
            residual = x
            attn_out, _ = self.mha_layers[i](x, x, x, need_weights=False)
            x = self.drop1(attn_out)
            x = self.ln1_layers[i](x + residual) # Add & Norm
            
            # 2. Feed Forward Network
            residual = x
            ffn_out = self.ffn_layers[i](x)
            x = self.drop2(ffn_out)
            x = self.ln2_layers[i](x + residual) # Add & Norm

        # --- MEAN POOLING & CLASSIFICATION ---
        pooled = x.mean(dim=1) 
        logits = self.classifier(pooled)
        return logits

# --- HELPER TO PREPARE DATA ---
def prepare_tensors(texts, vocab_size=VOCAB_SIZE, max_len=MAX_LEN):
    token = Tokenizer(num_words=vocab_size, oov_token="<unk>")
    token.fit_on_texts(texts)
    seqs = token.texts_to_sequences(texts)
    padded = pad_sequences(seqs, maxlen=max_len, padding='post')
    return torch.tensor(padded, dtype=torch.long), token