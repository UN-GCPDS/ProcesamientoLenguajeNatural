# %% [markdown]
# # Clase 8: El Puente Generativo: Decoding y Autoregresión
# 
# ## 1. Introducción a la Generación Autoregresiva
# 
# ### 1. Concepto de Autoregresión
# La generación autoregresiva es el paradigma fundamental detrás de modelos de lenguaje modernos como GPT. En este enfoque, la predicción del siguiente elemento en una secuencia depende de todos los elementos generados o dados anteriormente. Matemáticamente, modelamos la probabilidad conjunta de una secuencia de tokens $P(w_1, w_2, ..., w_n)$ como el producto de sus probabilidades condicionales:
# 
# $$ P(w_1, w_2, ..., w_n) = \prod_{i=1}^{n} P(w_i | w_1, ..., w_{i-1}) $$
# 
# En la práctica, esto significa que durante la inferencia, el modelo procesa una secuencia de entrada (prompt), predice un token en el paso de tiempo $t$, y este token predicho se concatena a la entrada original para convertirse en el nuevo input en el tiempo $t+1$. Este proceso recursivo y secuencial de retroalimentación es lo que da lugar al término "autoregresivo". Dado que cada paso depende de la salida del paso anterior, la generación no puede paralelizarse fácilmente.
# 
# ### 2. La Capa de Inferencia
# El paso final de la arquitectura de un Transformer Decoder es el responsable de la generación de texto. Después de que la secuencia de entrada atraviesa todos los bloques de atención (Self-Attention) y redes feed-forward, obtenemos una representación densa de la secuencia en el espacio latente, conocida como estados ocultos o *hidden states*. 
# 
# Para convertir estos estados ocultos nuevamente en palabras del lenguaje humano, empleamos la "Capa de Inferencia" (a menudo llamada "Language Modeling Head"). Esta es esencialmente una capa lineal definida por una matriz de pesos $W \in \mathbb{R}^{d_{model} \times |V|}$, donde $d_{model}$ es la dimensión del modelo y $|V|$ es el tamaño del vocabulario del tokenizer. Esta capa proyecta el estado oculto del último token de la secuencia en un vector del tamaño exacto del vocabulario. 
# 
# Los valores resultantes de esta proyección se llaman **logits** (puntuaciones brutas, no normalizadas). Finalmente, para darles una interpretación probabilística, aplicamos la función matemática Softmax, la cual convierte estos logits en una distribución de probabilidad válida donde todos los valores son positivos y suman 1. De esta distribución final es de donde muestrearemos o seleccionaremos el siguiente token.

# %%
# Instalación de dependencias necesarias (ejecutar en Colab)
# !pip install transformers torch

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configuramos la semilla para asegurar reproducibilidad en los experimentos
torch.manual_seed(42)

# %% [markdown]
# ## 2. Implementación del Bucle de Inferencia en PyTorch
# 
# ### 1. Carga del Modelo y Tokenizer
# Para ilustrar los conceptos de esta clase, utilizaremos `gpt2` (SmallGPT). Este es un modelo histórico y fundacional que mantiene la misma arquitectura base que modelos modernos gigantescos, pero con solo 124 millones de parámetros. Esto lo hace sumamente ligero y eficiente para correr de forma rápida en la GPU T4 que provee Google Colab de manera gratuita, evitando problemas de memoria (Out Of Memory - OOM).

# %%
# Configuración del dispositivo: Priorizamos GPU si está disponible
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Dispositivo de cómputo seleccionado: {device}")

# Seleccionamos el modelo a utilizar
model_name = "gpt2"

# Cargamos el Tokenizer (encargado de convertir texto a IDs y viceversa)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Cargamos el Modelo pre-entrenado y lo movemos a la GPU (o CPU)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

# Es fundamental poner el modelo en modo de evaluación para la inferencia.
# Esto desactiva comportamientos de entrenamiento como el Dropout.
model.eval()

# %% [markdown]
# ### 2. El Bucle de Generación Manual
# Aunque en la práctica diaria utilizaremos el método `.generate()` de la librería `transformers`, este método actúa como una "caja negra". Como ingenieros de Deep Learning, necesitamos programar este bucle manualmente para comprender a fondo la mecánica interna.
# 
# El siguiente código construye un bucle `while` para la inferencia. En cada iteración:
# 1. Se procesan los tokens actuales.
# 2. Se extraen los logits del vocabulario correspondientes al último estado.
# 3. Se seleccionan las probabilidades y el token de mayor valor (Greedy Search).
# 4. Se maneja el token especial `<EOS>` para saber cuándo detener la generación.
# 5. Se actualiza el tensor de `input_ids` agregando el nuevo token predicho.

# %%
def generate_text_manual(prompt_text, max_new_tokens=40):
    """
    Implementación desde cero de la generación autoregresiva usando estrategia Greedy.
    """
    # 1. Preparar el input inicial: Convertir texto a tensores y enviar a GPU
    input_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(device)
    
    print(f"Prompt recibido: '{prompt_text}'")
    print("-" * 50)
    
    # Identificamos el ID del token de fin de secuencia (End of Sequence)
    eos_token_id = tokenizer.eos_token_id
    
    tokens_generados = 0
    
    # Desactivamos el cálculo de gradientes para ahorrar memoria y acelerar el proceso
    with torch.no_grad(): 
        while tokens_generados < max_new_tokens:
            # 2. Paso Forward: Pasar la secuencia actual por el modelo
            outputs = model(input_ids)
            
            # 3. La salida tiene forma [batch_size, sequence_length, vocab_size]
            # Solo nos interesan los logits correspondientes al ÚLTIMO token ingresado
            last_token_logits = outputs.logits[0, -1, :] 
            
            # 4. Estrategia Greedy: Seleccionamos el índice del logit con mayor valor
            # Nota: El argmax del logit es idéntico al argmax de la probabilidad tras Softmax
            next_token_id = torch.argmax(last_token_logits)
            
            # Formateamos la dimensión para poder concatenarlo (hacerlo tamaño [1, 1])
            next_token_id_tensor = next_token_id.unsqueeze(0).unsqueeze(0)
            
            # 5. Condición de parada temprana: Verificar si el modelo predijo <EOS>
            if next_token_id.item() == eos_token_id:
                print("\n\n[INFO] El modelo generó el token <EOS>. Generación finalizada.")
                break
                
            # 6. Autoregresión: Concatenar el token predicho a la secuencia de entrada original
            input_ids = torch.cat([input_ids, next_token_id_tensor], dim=-1)
            
            # (Opcional) Visualización en tiempo real del token generado
            token_decodificado = tokenizer.decode(next_token_id)
            print(token_decodificado, end="", flush=True)
            
            tokens_generados += 1
            
    # Finalmente, decodificamos la secuencia completa finalizada
    texto_final = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return texto_final

# Ejecutamos el bucle manual
print("Iniciando generación iterativa manual...\n")
resultado_greedy = generate_text_manual("Artificial intelligence is transforming the world by")
print("\n\nResultado completo consolidado:\n", resultado_greedy)

# %% [markdown]
# ## 3. Estrategias de Decoding y Creatividad
# 
# ### 1. Greedy Search vs. Sampling
# En la sección anterior implementamos una estrategia llamada **Greedy Search** (búsqueda codiciosa). En ella, el modelo siempre toma el camino de mayor probabilidad en el paso actual sin mirar a largo plazo. Sus desventajas son claras: el texto resulta ser muy determinista, carece de creatividad y, a menudo, el modelo entra en bucles repetitivos de los que no puede escapar ("I think that I think that I think...").
# 
# La alternativa fundamental es el **Sampling** (muestreo probabilístico). En lugar de seleccionar siempre la palabra más probable con `argmax`, convertimos los logits en una distribución de probabilidades utilizando Softmax, y luego tiramos unos "dados ponderados" (multinomial) basados en esas probabilidades. Esto permite que el modelo explore caminos menos obvios, inyectando variabilidad y creatividad en el texto resultante.

# %%
def generate_with_sampling(prompt_text, max_new_tokens=40):
    """
    Generación utilizando Sampling puro (selección pseudo-aleatoria ponderada).
    """
    input_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(device)
    
    print(f"--- Sampling Puro ---")
    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(input_ids)
            last_logits = outputs.logits[0, -1, :]
            
            # 1. Transformar logits en probabilidades (suman 1)
            probabilities = F.softmax(last_logits, dim=-1)
            
            # 2. Sampling: Muestrear de la distribución usando torch.multinomial
            next_token_id = torch.multinomial(probabilities, num_samples=1).unsqueeze(0)
            
            if next_token_id.item() == tokenizer.eos_token_id:
                break
                
            input_ids = torch.cat([input_ids, next_token_id], dim=-1)
            
    return tokenizer.decode(input_ids[0])

# Demostración: El mismo prompt genera respuestas diferentes gracias al Sampling
prompt_test = "The secret to a happy life is"
print(f"Prompt: '{prompt_test}'\n")
print(f"Respuesta 1 (Sampling): \n{generate_with_sampling(prompt_test)}\n")
print(f"Respuesta 2 (Sampling, diferente recorrido): \n{generate_with_sampling(prompt_test)}")

# %% [markdown]
# ### 2. Hiperparámetros (Temperature, Top-K, Top-P)
# El sampling puro puede ser demasiado caótico; el modelo podría muestrear una palabra con probabilidad 0.001 que destruya la coherencia de la frase. Para controlar la distribución antes de hacer el muestreo, usamos hiperparámetros que actúan sobre los logits:
# 
# **1. Temperatura ($T$):** Divide los logits antes del Softmax: $p_i = \text{Softmax}(z_i / T)$.
# *   Si $T = 1.0$: No hay cambios, equivale al sampling puro.
# *   Si $T \to 0$ (ej. 0.1): La distribución se vuelve extremadamente "afilada". Las palabras probables se vuelven un 99% probables. Esto se aproxima al Greedy Search.
# *   Si $T > 1.0$ (ej. 1.5): Se "aplana" la distribución. Palabras menos probables ganan peso, aumentando drásticamente la creatividad a costa del sentido.
# 
# **2. Top-K:** Trunca la distribución. Filtra únicamente las $K$ palabras con los logits más altos. El resto del vocabulario ve sus probabilidades seteadas a 0. Esto asegura que el muestreo nunca escoja palabras absurdas de la "cola larga" de la distribución.
# 
# **3. Top-P (Nucleus Sampling):** En lugar de un valor $K$ estricto, suma acumulativamente las probabilidades ordenadas hasta alcanzar un umbral $P$ (ej. 0.90). El tamaño del subconjunto de tokens es dinámico: si la siguiente palabra es muy predecible, el "núcleo" tendrá 1 o 2 palabras. Si es incierta, el núcleo contendrá más palabras.

# %%
def generate_advanced_manual(prompt_text, temperature=1.0, top_k=0, max_new_tokens=40):
    """
    Bucle de inferencia incorporando Temperatura y Top-K.
    """
    input_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(input_ids)
            logits = outputs.logits[0, -1, :]
            
            # --- Modificación de la Distribución ---
            
            # 1. Aplicar Temperatura
            if temperature != 1.0:
                logits = logits / temperature
            
            # 2. Aplicar Top-K Filtering
            if top_k > 0:
                # Buscamos el k-ésimo logit más grande. topk() devuelve (valores, indices)
                # indices_to_remove será un tensor Booleano indicando qué logits son menores al k-ésimo.
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                
                # Asignamos un valor muy bajo (-Infinito) para que Softmax le dé 0% de probabilidad
                logits[indices_to_remove] = -float('Inf')
            
            # ---------------------------------------
            
            # Generar probabilidades finales y muestrear
            probabilities = F.softmax(logits, dim=-1)
            next_token_id = torch.multinomial(probabilities, num_samples=1).unsqueeze(0)
            
            if next_token_id.item() == tokenizer.eos_token_id:
                break
                
            input_ids = torch.cat([input_ids, next_token_id], dim=-1)
            
    return tokenizer.decode(input_ids[0])

# Experimento variando hiperparámetros
prompt_hp = "To build a time machine, you need"
print("--- Efecto de Temperatura y Top-K ---\n")

print("1. Temperatura Baja (T=0.3): Muy determinista, seguro y coherente.")
print(generate_advanced_manual(prompt_hp, temperature=0.3, max_new_tokens=30), "\n")

print("2. Temperatura Alta (T=2.0) sin Top-K: Probablemente ininteligible y caótico.")
print(generate_advanced_manual(prompt_hp, temperature=2.0, max_new_tokens=30), "\n")

print("3. Temperatura Alta (T=2.0) con Top-K=50: Alta creatividad pero mantiene estructura del lenguaje.")
print(generate_advanced_manual(prompt_hp, temperature=2.0, top_k=50, max_new_tokens=30))

# %% [markdown]
# ## 4. La Importancia del Prompting en la Respuesta
# 
# ### 1. El Prompt como Condicionamiento Probabilístico
# En el contexto del muestreo autoregresivo, el texto de entrada no es una simple instrucción para un agente inteligente; matemáticamente, actúa como un potente **condicionamiento estadístico**. 
# 
# Al suministrar un prompt, estamos restringiendo y guiando el espacio latente del Transformer. Las palabras iniciales dictaminan qué distribuciones de probabilidad se activarán a continuación. Si empezamos con lenguaje formal o de enciclopedia, las probabilidades marginales favorecerán un vocabulario académico. En cambio, formular el prompt como una pregunta de diálogo coloquial obligará al modelo a asignar puntuaciones altas (logits) a respuestas conversacionales, adaptando el registro. El "Prompt Engineering" a nivel fundamental se trata de buscar la configuración inicial que mueva la trayectoria predictiva del modelo hacia el espacio semántico deseado.
# 
# ### 2. Experimento Práctico
# A continuación, realizaremos un experimento empírico. Mantendremos el generador con parámetros constantes y una semilla manual fija. Solo cambiaremos sutilmente la redacción del prompt inicial para comprobar cómo la distribución condicional muta drásticamente ante un enfoque enciclopédico frente a un enfoque de diálogo/instruccional.

# %%
def test_prompting(prompt_text):
    """
    Función de testeo que fija la semilla pseudo-aleatoria para aislar la varianza.
    De esta forma, cualquier cambio en la respuesta proviene exclusivamente 
    del condicionamiento del prompt sobre las probabilidades subyacentes.
    """
    # Fijamos semilla para eliminar aleatoriedad entre pruebas
    torch.manual_seed(99) 
    
    print("-" * 60)
    print(f"PROMPT ORIGINAL: '{prompt_text}'")
    
    # Utilizamos Temperatura equilibrada y Top-K para buena calidad de texto
    output = generate_advanced_manual(prompt_text, temperature=0.7, top_k=50, max_new_tokens=40)
    
    # Extraemos y mostramos puramente el texto generado por el modelo
    texto_nuevo = output[len(prompt_text):].strip()
    print(f"GENERACIÓN: {texto_nuevo}")
    print("-" * 60)

print("\n--- Experimento de Condicionamiento del Prompt ---\n")

# Caso A: Condicionamiento Enciclopédico (Fact-based probability)
test_prompting("Quantum computing is formally defined as")

# Caso B: Condicionamiento de Diálogo Interactivo (Chatbot / Instruct probability)
test_prompting("User: What is quantum computing? Explain it to me simply.\nAI:")

# Caso C: Condicionamiento Creativo/Narrativo (Storytelling probability)
test_prompting("As the scientist turned on the quantum computer for the first time,")
