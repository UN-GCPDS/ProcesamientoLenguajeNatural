# %% [markdown]
## 1. El Dataset: Clasificación de SMS (Spam vs. Ham)

Para esta clase, utilizaremos el dataset **SMSSpamCollection**, un conjunto de datos clásico en el procesamiento de lenguaje natural (NLP). El objetivo es construir modelos capaces de distinguir entre mensajes legítimos (**ham**) y mensajes no deseados (**spam**).

A diferencia de la sesión anterior donde trabajamos con reseñas de películas, los SMS presentan retos distintos: son textos cortos, con jerga y a menudo con errores ortográficos.

### 1.1 Carga y Preprocesamiento
Iniciaremos cargando los datos y realizando un mapeo numérico: **ham** será representado por el valor **0** y **spam** por el **1**. Además, dividiremos el conjunto en un 80% para entrenamiento y un 20% para pruebas, manteniendo la proporción de clases mediante la estratificación.

# %%
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from collections import Counter

# Configuración de hiperparámetros de datos
MAX_LENGTH = 128
BATCH_SIZE = 64

# 1. Carga de datos desde la fuente oficial
url = "[https://raw.githubusercontent.com/juacardonahe/Curso_NLP/refs/heads/main/data/SMSSpamCollection/SMSSpamCollection](https://raw.githubusercontent.com/juacardonahe/Curso_NLP/refs/heads/main/data/SMSSpamCollection/SMSSpamCollection)"
df = pd.read_csv(url, sep="\t", header=None, names=["label", "message"])

# 2. Mapeo de etiquetas numéricas
df["label"] = df["label"].map({"ham": 0, "spam": 1})

# 3. Split de entrenamiento y prueba (80/20)
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    df["message"], 
    df["label"], 
    test_size=0.2, 
    random_state=42, 
    stratify=df["label"]
)

# %% [markdown]
### 1.2 Tokenización y Vocabulario "Desde Cero"
En PyTorch, es una excelente práctica pedagógica construir un vocabulario básico para entender cómo los textos se convierten en índices numéricos antes de pasar a herramientas automáticas.

# %%
class SimpleVocabulary:
    def __init__(self, messages):
        # Contamos frecuencias y creamos el diccionario
        self.word_counts = Counter(" ".join(messages).lower().split())
        # Reservamos el 0 para [PAD] y el 1 para [UNK] (desconocidos)
        self.vocab = {"[PAD]": 0, "[UNK]": 1}
        for word, _ in self.word_counts.items():
            self.vocab[word] = len(self.vocab)
        self.id_to_word = {v: k for k, v in self.vocab.items()}

    def encode(self, text, max_len):
        tokens = text.lower().split()
        ids = [self.vocab.get(token, 1) for token in tokens]
        # Truncado y Padding manual
        ids = ids[:max_len]
        ids += [0] * (max_len - len(ids))
        return torch.tensor(ids)

# Instanciamos el vocabulario con el set de entrenamiento
vocab = SimpleVocabulary(X_train_raw)
vocab_size = len(vocab.vocab)
print(f"Tamaño del Vocabulario: {vocab_size}")

# %% [markdown]
### 1.3 El PyTorch Dataset
Para alimentar nuestras arquitecturas Transformer, encapsulamos los datos en un objeto `Dataset`.

# %%
class SMSDataset(Dataset):
    def __init__(self, messages, labels, vocab, max_len):
        self.messages = messages.tolist()
        self.labels = labels.tolist()
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.messages)

    def __getitem__(self, idx):
        return self.vocab.encode(self.messages[idx], self.max_len), torch.tensor(self.labels[idx], dtype=torch.float32)

# Creación de los DataLoaders
train_ds = SMSDataset(X_train_raw, y_train, vocab, MAX_LENGTH)
test_ds = SMSDataset(X_test_raw, y_test, vocab, MAX_LENGTH)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

# %% [markdown]
## 2. Sabor 1: Encoder-Only (El enfoque "BERT")

En esta arquitectura, un **encoder bidireccional** procesa toda la secuencia simultáneamente utilizando *self-attention* no causal. Esto permite que cada palabra tenga acceso a toda la información de la oración, tanto a su izquierda como a su derecha, lo cual es ideal para tareas de comprensión y discriminación.

### 2.1 ¿Por qué usar solo el Encoder?
*   **Tareas Discriminativas:** Las tareas como clasificación de texto (Spam/Ham), reconocimiento de entidades (NER) o búsqueda semántica suelen obtener mejores resultados con encoders (ej. BERT, RoBERTa).
*   **Contexto Global:** Al no tener una máscara causal, el modelo puede "mirar" el mensaje completo antes de decidir si es spam.



### 2.2 Estrategia de Pooling
Como el Transformer entrega un vector por cada palabra de entrada, necesitamos una forma de reducir esa secuencia a un único vector representativo para el clasificador final. En este ejemplo, implementaremos el **Global Average Pooling**, promediando únicamente los vectores de los tokens válidos (ignorando el padding).

# %%
import torch.nn as nn
import torch.nn.functional as F

class EncoderOnlyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, num_heads=4, num_layers=3, d_ff=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=MAX_LENGTH) # Reutilizamos de la Clase 1
        
        # Usamos la implementación de PyTorch para el bloque Encoder
        # norm_first=True implementa el esquema "Pre-LN" mencionado en el material
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=num_heads, 
            dim_feedforward=d_ff, 
            dropout=dropout,
            activation='gelu', # GELU es el estándar en encoders modernos
            batch_first=True,
            norm_first=True 
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.classifier = nn.Linear(d_model, 1) # Salida binaria para Spam/Ham
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # 1. Padding Mask: True para los valores que NO son padding (0)
        # PyTorch espera True en las posiciones que DEBEN ignorarse
        padding_mask = (x == 0)
        
        # 2. Embeddings + Positional Encoding
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        
        # 3. Procesamiento por los bloques Encoder
        x = self.transformer_encoder(x, src_key_padding_mask=padding_mask)
        x = self.layer_norm(x)
        
        # 4. Pooling: Promedio de tokens válidos (Global Average Pooling)
        # Creamos una máscara (B, T, 1) para multiplicar y promediar
        mask_f = (~padding_mask).unsqueeze(-1).float()
        sum_x = torch.sum(x * mask_f, dim=1)
        len_x = torch.sum(mask_f, dim=1) + 1e-9
        pooled = sum_x / len_x # (Batch, d_model)
        
        # 5. Clasificación final con sigmoide
        logits = self.classifier(pooled)
        return torch.sigmoid(logits)

# %% [markdown]
### 2.3 Entrenamiento y Evaluación
Entrenaremos el modelo utilizando **Binary Cross Entropy** y evaluaremos con **Accuracy** y **AUC**, tal como se sugiere en la metodología compartida.

# %%
from sklearn.metrics import accuracy_score, roc_auc_score

def train_and_eval(model, train_loader, test_loader, epochs=3):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCELoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for texts, labels in train_loader:
            texts, labels = texts.to(device), labels.to(device).unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Época {epoch+1}/{epochs} | Pérdida: {total_loss/len(train_loader):.4f}")

    # Evaluación Final
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for texts, labels in test_loader:
            texts = texts.to(device)
            outputs = model(texts)
            all_probs.extend(outputs.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    probs = np.array(all_probs).ravel()
    preds = (probs >= 0.5).astype(int)
    print(f"\nEncoder-Only | Acc: {accuracy_score(all_labels, preds):.4f} | AUC: {roc_auc_score(all_labels, probs):.4f}")

# Instanciamos y entrenamos
model_encoder = EncoderOnlyTransformer(vocab_size)
train_and_eval(model_encoder, train_loader, test_loader)

# %% [markdown]
## 3. Sabor 2: Decoder-Only (El enfoque "GPT")

En esta arquitectura, el modelo procesa la información de forma **autorregresiva**. A diferencia del Encoder, aquí cada posición solo tiene permitido "mirar" hacia el pasado y hacia sí misma. 

### 3.1 La Máscara Causal: La Tiranía del Tiempo
Para evitar que el modelo haga "trampa" viendo las palabras que siguen en la oración, aplicamos una **máscara causal triangular**. Esta máscara asegura que la atención en la posición $i$ solo dependa de las posiciones $j \le i$.



*   **Generación vs. Clasificación:** Aunque los Decoders (como GPT) son ideales para **generación de texto**, también pueden adaptarse para **clasificación**.
*   **Representación del Último Token:** En lugar de promediar toda la secuencia, es común tomar la representación del **último token válido** (el último antes del padding) para realizar la predicción, ya que, por la naturaleza de la máscara, ese token es el único que ha "visto" a todos sus predecesores.

# %%
class DecoderOnlyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, num_heads=4, num_layers=3, d_ff=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=MAX_LENGTH)
        
        # Implementación del bloque Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        self.classifier = nn.Linear(d_model, 1)
        self.layer_norm = nn.LayerNorm(d_model)

    def generate_causal_mask(self, sz):
        # Genera una matriz triangular superior llena de -inf
        mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
        return mask

    def forward(self, x):
        batch_size, seq_len = x.shape
        device = x.device
        
        # 1. Máscaras: Causal (T, T) y Padding (B, T)
        causal_mask = self.generate_causal_mask(seq_len).to(device)
        padding_mask = (x == 0)
        
        # 2. Embeddings + Posición
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        
        # 3. Procesamiento Decoder (En Decoder-Only, tgt y memory son lo mismo o memory se ignora)
        # Aquí pasamos 'x' como 'tgt'. Al ser Decoder-only puro, no hay cross-attention con un encoder.
        x = self.transformer_decoder(tgt=x, memory=x, tgt_mask=causal_mask, tgt_key_padding_mask=padding_mask)
        x = self.layer_norm(x)
        
        # 4. Selección del ÚLTIMO token válido
        # Calculamos la longitud de cada secuencia para hallar el índice final
        lengths = torch.sum(~padding_mask, dim=1) - 1 # (Batch,)
        idx = torch.clamp(lengths, min=0).unsqueeze(1).unsqueeze(2).expand(-1, -1, self.d_model)
        last_token = torch.gather(x, 1, idx).squeeze(1) # (Batch, d_model)
        
        # 5. Clasificación
        return torch.sigmoid(self.classifier(last_token))

# %% [markdown]
### 3.2 Entrenamiento del Decoder
Aunque la arquitectura cambia, el protocolo de entrenamiento se mantiene igual para permitir una comparación justa entre "sabores".

# %%
# Instanciamos y entrenamos el sabor Decoder-Only
model_decoder = DecoderOnlyTransformer(vocab_size)
print("Entrenando Decoder-Only...")
train_and_eval(model_decoder, train_loader, test_loader)

# %% [markdown]
## 4. Sabor 3: Encoder-Decoder (El enfoque "seq2seq")

Esta es la arquitectura completa y original del Transformer propuesto en "Attention is All You Need". Combina la capacidad de comprensión del **Encoder** con la capacidad de generación del **Decoder**, conectados por un puente fundamental: la **Cross-Attention**.

### 4.1 Cross-Attention: El Puente de Información
A diferencia de la *Self-Attention*, donde las consultas (Q), llaves (K) y valores (V) provienen de la misma secuencia, en la **Cross-Attention**:
*   **Query (Q):** Proviene del Decoder (lo que se está generando o clasificando en ese momento).
*   **Key (K) y Value (V):** Provienen de la salida del Encoder (el contexto completo del mensaje original).



### 4.2 Clasificación en un Solo Paso
Para tareas de clasificación con esta arquitectura, como los modelos **T5 o BART**, se suele alimentar el Decoder con un **token de inicio especial** (BOS - Beginning of Sentence). El Decoder utiliza este token para "preguntar" al contexto del Encoder qué tipo de mensaje es, y realizamos la predicción basada en la representación de ese único paso.

# %%
class EncoderDecoderTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, num_heads=4, num_layers=2, d_ff=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        # Compartimos el embedding entre encoder y decoder para simplificar
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=MAX_LENGTH)
        
        # Arquitectura completa de PyTorch
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True # Esquema Pre-LN vital para estabilidad
        )
        
        self.classifier = nn.Linear(d_model, 1)

    def forward(self, src, tgt):
        # 1. Máscaras de Padding
        src_padding_mask = (src == 0)
        tgt_padding_mask = (tgt == 0)
        
        # 2. Máscara Causal para el Decoder
        tgt_seq_len = tgt.shape[1]
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_seq_len).to(src.device)
        
        # 3. Embeddings + Posición
        src_emb = self.pos_encoding(self.embedding(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_encoding(self.embedding(tgt) * math.sqrt(self.d_model))
        
        # 4. Flujo Seq2Seq completo
        # src_emb -> Encoder -> memory
        # tgt_emb + memory -> Decoder -> out
        out = self.transformer(
            src=src_emb, 
            tgt=tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask
        )
        
        # 5. Tomamos la representación del token BOS (primer paso del decoder)
        # Para clasificación de un paso, y_cls es la salida de la posición 0
        y_cls = out[:, 0, :] 
        
        return torch.sigmoid(self.classifier(y_cls))

# %% [markdown]
### 4.3 Entrenamiento con Token BOS
Para el Decoder, necesitamos un "disparador". Usaremos un vector de tokens constantes (índice 1) para representar el inicio de la secuencia.

# %%
def train_and_eval_seq2seq(model, train_loader, test_loader, epochs=3):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCELoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Creamos el token BOS (Beginning Of Sentence)
    # En este caso, usamos un tensor de unos de longitud 1 para cada mensaje
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for texts, labels in train_loader:
            texts, labels = texts.to(device), labels.to(device).unsqueeze(1)
            # El decoder recibe el token de inicio (BOS)
            bos_token = torch.ones((texts.size(0), 1), dtype=torch.long).to(device)
            
            optimizer.zero_grad()
            outputs = model(texts, bos_token)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Época {epoch+1}/{epochs} | Pérdida: {total_loss/len(train_loader):.4f}")

    # Evaluación
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for texts, labels in test_loader:
            texts = texts.to(device)
            bos_token = torch.ones((texts.size(0), 1), dtype=torch.long).to(device)
            outputs = model(texts, bos_token)
            all_probs.extend(outputs.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    probs = np.array(all_probs).ravel()
    preds = (probs >= 0.5).astype(int)
    print(f"\nEnc-Dec | Acc: {accuracy_score(all_labels, preds):.4f} | AUC: {roc_auc_score(all_labels, probs):.4f}")

# Instanciamos y entrenamos el sabor completo
model_encdec = EncoderDecoderTransformer(vocab_size)
print("Entrenando Encoder-Decoder...")
train_and_eval_seq2seq(model_encdec, train_loader, test_loader)

---

# %% [markdown]
## 5. Comparativa Final y Toma de Decisiones

Hemos explorado las tres "personalidades" del Transformer. Aunque todas comparten el mecanismo de atención, la forma en que restringen el flujo de información (máscaras) y cómo conectan sus bloques determina su especialidad.

### 5.1 Tabla de Decisión Arquitectónica

Esta tabla resume las fortalezas y casos de uso que hemos implementado hoy:

| Arquitectura | Enfoque de Máscara | Fortalezas | Modelos Famosos | Tareas Típicas |
| :--- | :--- | :--- | :--- | :--- |
| **Encoder-Only** | **Padding** (Bidireccional) | Comprensión profunda y contexto global | BERT, RoBERTa | Clasificación, NER, Búsqueda semántica |
| **Decoder-Only** | **Causal** (Unidireccional) | Generación fluida y coherencia temporal | GPT-4, LLaMA | Chatbots, Autocompletado, Storytelling |
| **Encoder-Decoder**| **Mixto** (Self + Cross Attn) | Mapeo complejo de entrada a salida | T5, BART | Traducción, Resumen condicional |



---

### 5.2 Análisis de Resultados (SMS Spam)

Al observar las métricas obtenidas en las secciones anteriores, podemos notar ciertos patrones:

*   **Encoder-Only:** Suele obtener el mejor **Accuracy** y **AUC** en esta tarea de SMS. Al ser bidireccional, "entiende" mejor el mensaje corto sin importar el orden de las palabras clave.
*   **Decoder-Only:** Funciona bien, pero es más "sensible" a la posición de las palabras. Si la palabra que delata el Spam está al principio, el modelo debe recordarla a través de muchas capas hasta llegar al último token.
*   **Encoder-Decoder:** Es la arquitectura más pesada y costosa de entrenar. Para una tarea simple de clasificación, puede ser "demasiada artillería", aunque su versatilidad es inigualable en tareas *text-to-text*.

---