# %% [markdown]
## 7. Ejercicio Final: Clasificación Real y Visualización de Atención (XAI)

Para que este ejercicio sea genuino, redefiniremos nuestro modelo para que sea "transparente", es decir, que nos permita extraer la matriz de atención de la última capa. Entrenaremos el modelo con una porción del dataset IMDb para que aprenda a reconocer palabras clave.

### 7.1 Redefinición del Modelo para Extracción de Pesos
> *Nota: Modificamos el `forward` de las clases para que retornen los pesos de atención.*

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttentionTransparent(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        q = self.w_q(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Cálculo de Scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Guardamos los pesos de atención
        attn_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        
        return self.w_o(output), attn_weights # Retornamos los pesos reales

class TransformerBlockTransparent(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttentionTransparent(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model))
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        attn_out, weights = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x, weights # Pasamos los pesos hacia arriba

class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=128)
        self.layers = nn.ModuleList([TransformerBlockTransparent(d_model, num_heads, d_model*4) for _ in range(num_layers)])
        self.classifier = nn.Linear(d_model, num_classes)
        
    def forward(self, x, mask=None):
        x = self.pos_encoding(self.embedding(x))
        last_weights = None
        for layer in self.layers:
            x, last_weights = layer(x, mask)
        
        # Tomamos el promedio (Pooling) para clasificar
        pooled = torch.mean(x, dim=1)
        return self.classifier(pooled), last_weights

# %% [markdown]
### 7.2 Entrenamiento Breve con IMDb
Para que el heatmap no sea ruido, el modelo debe "aprender" a ignorar el padding y enfocarse en palabras clave.

# %%
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

# 1. Cargar subconjunto pequeño para rapidez (2 epochs en 2000 ejemplos)
dataset = load_dataset("imdb", split={'train': 'train[:2000]', 'test': 'test[:500]'})
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_fn(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

tokenized_data = dataset.map(tokenize_fn, batched=True)
tokenized_data.set_format(type='torch', columns=['input_ids', 'label'])
train_loader = DataLoader(tokenized_data['train'], batch_size=16, shuffle=True)

# 2. Configurar Entrenamiento
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerClassifier(tokenizer.vocab_size, d_model=128, num_heads=4, num_layers=2, num_classes=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

# 3. Entrenamiento rápido (XAI requiere que el modelo sepa qué está haciendo)
model.train()
print("Entrenando modelo para ajustar pesos de atención...")
for epoch in range(2): # 2 épocas son suficientes para ver patrones de atención
    for batch in train_loader:
        optimizer.zero_grad()
        ids = batch['input_ids'].to(device)
        mask = (ids != tokenizer.pad_token_id).unsqueeze(1).unsqueeze(2) # Padding Mask real
        labels = batch['label'].to(device)
        
        logits, _ = model(ids, mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
    print(f"Época {epoch+1} completada.")

# %% [markdown]
### 7.3 Visualización XAI (Real)
Ahora extraeremos la matriz de atención **auténtica** del modelo entrenado.

# %%
import seaborn as sns
import matplotlib.pyplot as plt

def plot_real_attention(model, sentence, tokenizer, device):
    model.eval()
    inputs = tokenizer(sentence, return_tensors="pt", padding="max_length", truncation=True, max_length=24).to(device)
    ids = inputs['input_ids']
    mask = (ids != tokenizer.pad_token_id).unsqueeze(1).unsqueeze(2)
    
    with torch.no_grad():
        logits, attn_weights = model(ids, mask)
        prediction = torch.argmax(logits, dim=-1).item()
    
    # attn_weights shape: (batch, heads, seq_len, seq_len)
    # Promediamos las cabezas de atención para ver el enfoque global
    avg_attention = attn_weights[0].mean(dim=0).cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(ids[0])

    plt.figure(figsize=(10, 8))
    sns.heatmap(avg_attention, xticklabels=tokens, yticklabels=tokens, cmap='viridis')
    plt.title(f"XAI: Atención Real - Predicción: {'Positiva' if prediction == 1 else 'Negativa'}")
    plt.show()

# Probamos con una frase cargada de sentimiento
test_sentence = "The movie was an absolute masterpiece, truly incredible acting."
plot_real_attention(model, test_sentence, tokenizer, device)

# %% [markdown]
### ¿Qué deben observar los estudiantes ahora?
1.  **Estructura de Tokens:** Verán `[CLS]` al inicio y `[SEP]` al final, con `[PAD]` al final de la secuencia.
2.  **Efecto de la Máscara:** Los tokens `[PAD]` ahora deberían verse **púrpura oscuro** (valor 0), porque usamos la `Padding Mask` real durante el entrenamiento. El modelo ha aprendido que esos tokens no contienen información.
3.  **Foco Semántico:** Las palabras "masterpiece" e "incredible" deberían tener líneas o puntos más claros (amarillo/verde), indicando que el modelo "atendió" a esas palabras para clasificar la reseña como positiva.
4.  **Token [CLS]:** Notarán que muchas palabras "miran" al token `[CLS]`, ya que es allí donde se consolida la decisión de clasificación.
