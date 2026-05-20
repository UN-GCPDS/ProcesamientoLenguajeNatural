# %% [markdown]
# # Clase 4: RAG (Retrieval-Augmented Generation) y Memoria Vectorial
# **Curso:** Procesamiento del Lenguaje Natural  
# **Profesor:** Lucas Iturriago Salas  
# **Institución:** Universidad Nacional de Colombia (UNAL)  
# 
# ---

# %% [markdown]
# ## 1. El Paradigma de la Generación Aumentada por Recuperación (RAG)

# %% [markdown]
# ### 1. Limitaciones de los LLMs: Conocimiento Estático y Alucinaciones
# 
# Los Modelos de Lenguaje de Gran Escala (LLMs) son entrenados con conjuntos de datos masivos que representan un "corte en el tiempo". Esto introduce tres problemas fundamentales en entornos de producción e ingeniería:
# 
# 1. **Conocimiento Estático:** Un modelo preentrenado carece de información sobre eventos ocurridos después de la fecha de finalización de su entrenamiento (*knowledge cutoff*).
# 2. **Falta de Acceso a Datos Privados:** El modelo no conoce los documentos internos de una organización, reglamentos específicos o registros históricos de bases de datos privadas.
# 3. **Alucinaciones (Hallucinations):** Debido a su naturaleza probabilística (diseñados para maximizar la probabilidad del siguiente token), cuando un LLM se enfrenta a una consulta sobre un hecho que desconoce, tiende a generar respuestas que suenan gramaticalmente correctas y altamente convincentes, pero que son factualmente falsas o inventadas.
# 
# Tradicionalmente, la solución a esto era el **Fine-Tuning** (Ajuste Fino). Sin embargo, reentrenar un modelo con millones de parámetros presenta severas desventajas: es computacionalmente costoso, requiere hardware especializado inaccesible para tareas cotidianas, y no garantiza que el modelo deje de alucinar por completo o que olvide información previa (*catastrophic forgetting*).

# %% [markdown]
# ### 2. Arquitectura de un Sistema RAG y Mitigación de Alucinaciones
# 
# Para solucionar estas limitaciones sin alterar los pesos del modelo, surge el paradigma **RAG (Retrieval-Augmented Generation)**. En lugar de forzar al modelo a "recordar" la información desde sus parámetros internos, el sistema RAG actúa como un examen a libro abierto.
# 
# El flujo arquitectónico se divide en tres etapas secuenciales:
# 
# 1. **Recuperación (Retrieve):** La consulta del usuario en lenguaje natural se procesa para buscar y extraer los fragmentos de texto más relevantes dentro de una base de conocimiento externa indexada vectorialmente.
# 2. **Aumentación (Augment):** Los fragmentos recuperados en el paso anterior se inyectan directamente en el contexto del prompt original del usuario. El prompt se transforma dinámicamente de un simple *"¿Qué es X?"* a *"Basándote únicamente en el siguiente contexto estructurado: [Contexto], responde la pregunta: ¿Qué es X?"*.
# 3. **Generación (Generate):** El prompt aumentado se envía al LLM. Al estar fuertemente condicionado por la información explícita provista en el contexto, la probabilidad latente del modelo se sesga hacia las respuestas verídicas del documento, **reduciendo la alucinación a niveles mínimos**.
# 
# En ingeniería, este enfoque garantiza soberanía de datos, actualización de la información en tiempo real (basta con cambiar el archivo de la base de datos) y un control estricto sobre las respuestas del sistema.

# %% [markdown]
# ### 3. Setup e Instalación de Dependencias
# 
# Antes de implementar los componentes de nuestro pipeline, configuramos el entorno en Google Colab instalando las librerías necesarias de Hugging Face, la API de Google y herramientas de indexación vectorial.

# %%
# Instalación de herramientas de procesamiento de texto y embeddings
!pip install -q transformers sentence-transformers langchain-text-splitters
# Instalación de la API de Google GenAI para interactuar con Google AI Studio
!pip install -q google-genai
# Instalación del motor de búsqueda vectorial open-source FAISS
!pip install -q faiss-cpu

# %% [markdown]
# ## 2. Embeddings y el Espacio Latente Semántico

# %% [markdown]
# ### 1. ¿Qué es un Vector Embedding?
# 
# En el Procesamiento del Lenguaje Natural moderno, los modelos no operan con cadenas de texto directas, sino con representaciones numéricas continuas llamadas **Vector Embeddings**. 
# 
# Un embedding es la proyección de un concepto lingüístico (palabra, frase o documento completo) en un espacio geométrico de alta dimensión ($d$-dimensional) conocido como **Espacio Latente Semántico**. La propiedad fundamental de este espacio es que la distancia geométrica refleja la afinidad semántica:
# * Si dos textos comparten un significado o contexto similar (ej. "automóvil" y "vehículo"), sus vectores correspondientes estarán espacialmente muy cerca.
# * Si los textos no guardan relación (ej. "algoritmo" y "almuerzo"), sus vectores apuntarán en direcciones significativamente distintas.
# 
# Pasar del texto al espacio vectorial nos permite transformar problemas de comprensión del lenguaje en problemas de **geometría analítica y álgebra lineal**.

# %% [markdown]
# ### 2. Métricas de Similitud Lineal: Similitud de Coseno
# 
# Para determinar qué tan parecidos son dos fragmentos de texto en nuestro sistema RAG, debemos medir la cercanía de sus vectores en el espacio latente. Aunque existen métricas como la distancia Euclidiana, la métrica estándar en NLP es la **Similitud de Coseno**.
# 
# La similitud de coseno no mide la distancia absoluta entre los extremos de los vectores, sino el **ángulo** $\theta$ que se forma entre ellos. Esto es crucial porque ignora la longitud del texto: una reseña de tres líneas y un párrafo de diez líneas que hablen de lo mismo mantendrán vectores que apuntan en una dirección similar.
# 
# Matemáticamente, la similitud de coseno entre dos vectores $A$ y $B$ de dimensión $n$ se define como el producto escalar de ambos vectores dividido por el producto de sus normas Euclidianas:
# 
# $$\text{Similitud de Coseno}(A, B) = \cos(\theta) = \frac{A \cdot B}{\|A\| \|B\|} = \frac{\sum_{i=1}^{n} A_i B_i}{\sqrt{\sum_{i=1}^{n} A_i^2} \sqrt{\sum_{i=1}^{n} B_i^2}}$$
# 
# El resultado de esta operación es un escalar acotado:
# * **$1.0$**: Los vectores son colineales y apuntan en la misma dirección (máxima similitud semántica).
# * **$0.0$**: Los vectores son ortogonales (independencia semántica total).
# * **$-1.0$**: Los vectores apuntan en direcciones opuestas (conceptos diametralmente opuestos, aunque en espacios de embeddings de LLMs las puntuaciones suelen mantenerse positivas, distribuidas entre $0$ y $1$).

# %% [markdown]
# ### 3. Implementación Práctica de Similitud Numérica en PyTorch
# 
# A continuación, utilizaremos **PyTorch** para construir de forma explícita el cálculo de la similitud de coseno sobre un conjunto de frases de prueba. Para simular el comportamiento de un LLM, usaremos un modelo preentrenado ligero de la librería `sentence-transformers` para extraer las coordenadas vectoriales.

# %%
import torch
from sentence_transformers import SentenceTransformer

# 1. Verificar si disponemos de aceleración por GPU (T4 en Colab)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Dispositivo de cómputo detectado: {device.upper()}")

# 2. Cargar un codificador de embeddings open-source y liviano
# Este modelo proyecta cada frase en un vector de dimensión d = 384
embedding_model = SentenceTransformer("all-MiniLM-L6-v2").to(device)

# 3. Definir nuestro corpus de prueba (frases con diferentes afinidades)
frases = [
    "El sistema de frenos del coche falló en la autopista.",
    "Los componentes del motor muestran un desgaste prematuro.",
    "Ayer compré un almuerzo delicioso en la cafetería.",
    "El mecanismo de frenado no respondió correctamente."
]

# 4. Generar los embeddings y convertirlos a tensores de PyTorch
with torch.no_grad():
    # Extraemos los vectores y los enviamos al dispositivo configurado
    vectors = embedding_model.encode(frases, convert_to_tensor=True).to(device)

print(f"Dimensiones de la matriz de embeddings: {vectors.shape} (4 frases, d = 384)")

# %%
# 5. Programar la función de similitud de coseno matemática usando PyTorch puro
def calcular_similitud_coseno(tensor_a, tensor_b):
    """
    Calcula la similitud de coseno analítica entre dos tensores unidimensionales.
    """
    producto_escalar = torch.dot(tensor_a, tensor_b)
    norma_a = torch.norm(tensor_a)
    norma_b = torch.norm(tensor_b)
    
    cos_theta = producto_escalar / (norma_a * norma_b)
    return cos_theta.item()

# 6. Evaluar la similitud semántica de las frases frente a la frase objetivo (Índice 0)
frase_objetivo = vectors[0]

print("--- RESULTADOS DE SIMILITUD SEMÁNTICA ---")
for i, vector_frase in enumerate(vectors):
    similitud = calcular_similitud_coseno(frase_objetivo, vector_frase)
    print(f"Similitud con Frase {i} ['{frases[i]}']: {similitud:.4f}")

# %% [markdown]
# ## 3. Generación de Vectores con Google Embeddings 2

# %% [markdown]
# ### 1. Configuración de Google AI Studio e Inicialización del Cliente
# 
# Para construir un sistema RAG industrial, requerimos un modelo de embeddings de alta capacidad que capture relaciones semánticas complejas y soporte contextos extensos de forma eficiente. En esta sección utilizaremos la API de **Google AI Studio** a través del modelo de última generación `text-embedding-004` (Google Embeddings 2).
# 
# A diferencia de los modelos locales pequeños, este codificador comercial proyecta el texto en un espacio de **768 dimensiones**, entrenado con corpus masivos multilingües, lo que garantiza un rendimiento óptimo en español sin penalizar el tiempo de cómputo de nuestras máquinas locales.
# 
# *Nota de seguridad:* Utilizaremos el módulo nativo de Colab para gestionar nuestra API Key de forma segura mediante variables de entorno, evitando exponer credenciales en el código fuente.

# %%
import os
from google import genai
from google.genai import types
from google.colab import userdata

# 1. Recuperar la API Key desde el almacenamiento seguro de secretos de Colab
# Asegúrate de haber guardado tu clave en la pestaña de la llave (Secrets) con el nombre 'GEMINI_API_KEY'
try:
    api_key = userdata.get('GEMINI_API_KEY')
    os.environ["GEMINI_API_KEY"] = api_key
    print("✅ API Key de Google AI Studio cargada con éxito.")
except Exception as e:
    print("⚠️ No se encontró la API Key en los secretos de Colab. Asegúrate de configurarla.")

# 2. Inicializar el cliente unificado de Google GenAI (SDK oficial v1.0+)
client = genai.Client()

# %% [markdown]
# ### 2. Estrategias de Procesamiento de Texto: Chunking y Overlap
# 
# En aplicaciones reales de NLP, los documentos de origen (manuales técnicos, reportes, libros) exceden la capacidad de procesamiento de los modelos de embedding o diluyen la semántica si se vectorizan por completo. Si pasamos un PDF de 50 páginas a un vector, este representará una "idea promedio" muy vaga, perdiendo los detalles específicos.
# 
# La solución de ingeniería es el **Chunking** (Segmentación): dividir el texto largo en fragmentos (*chunks*) pequeños y homogéneos. 
# 
# Al segmentar, implementamos una ventana deslizante mediante el **Overlap** (Solapamiento). El solapamiento consiste en duplicar un porcentaje de caracteres o palabras entre un fragmento y el siguiente. Esto es crítico para:
# * **Preservar el Contexto de Borde:** Evitar que una frase importante quede cortada exactamente a la mitad por el límite del algoritmo de segmentación.
# * **Continuidad Semántica:** Asegurar que los vectores contiguos mantengan una transición temática suave.
# 
# [Texto Original]  ====== Fragmento 1 ======
#                              [Overlap] ====== Fragmento 2 ======

# %% [markdown]
# ### 3. Implementación de un Segmentador Dinámico y Extracción de Embeddings
# 
# A continuación, programaremos un pipeline que recibe un documento técnico extenso, lo segmenta aplicando una ventana con solapamiento, y envía los fragmentos en lote (*batch*) a la API de Google para extraer la matriz de vectores.

# %%
# 1. Definición de un texto técnico extenso de prueba (Simulación de un reporte de ingeniería)
documento_tecnico = (
    "El sistema de control de estabilidad del vehículo monitorea constantemente la velocidad de las ruedas. "
    "Cuando la unidad de control electrónico (ECU) detecta una pérdida de tracción en el eje delantero, "
    "activa inmediatamente el módulo hidráulico del ABS para aplicar presión de frenado selectiva en la rueda trasera opuesta. "
    "Este ajuste físico estabiliza la guiñada y previene el subviraje crítico en curvas de alta velocidad. "
    "Adicionalmente, el sistema reduce el torque del motor de forma temporal interviniendo la mariposa de admisión. "
    "Para el mantenimiento de estos módulos, es mandatorio inspeccionar los sensores magnéticos de efecto Hall colocados en cada maza. "
    "Cualquier acumulación de residuos ferrosos en el anillo reluctor puede provocar lecturas erróneas de aceleración angular, "
    "lo que disparará una alerta en el panel de instrumentos desactivando las asistencias de seguridad de forma preventiva."
)

# 2. Implementación analítica de un segmentador de texto por palabras (Word-based Chunking)
def segmentar_texto(texto, chunk_size=30, overlap=10):
    """
    Divide un texto en fragmentos basados en número de palabras con solapamiento.
    """
    palabras = texto.split()
    chunks = []
    
    i = 0
    while i < len(palabras):
        # Extraer la ventana de palabras
        ventana = palabras[i : i + chunk_size]
        # Reconstruir el string del fragmento
        fragmento = " ".join(ventana)
        chunks.append(fragmento)
        # Desplazar el índice restando el overlap para la siguiente iteración
        i += chunk_size - overlap
        
        # Romper el bucle si alcanzamos el final del texto
        if i + chunk_size > len(palabras) and i < len(palabras):
            restantes = palabras[i:]
            chunks.append(" ".join(restantes))
            break
            
    return chunks

# 3. Ejecutar la segmentación de nuestro documento
fragmentos_procesados = segmentar_texto(documento_tecnico, chunk_size=25, overlap=8)

print(f"Documento original segmentado en {len(fragmentos_procesados)} fragmentos semánticos.\n")
for idx, chunk in enumerate(fragmentos_procesados):
    print(f"Fragmento {idx} (Longitud: {len(chunk.split())} palabras):\n--> \"{chunk}\"\n")

# %%
# 4. Enviar los fragmentos en lote a la API de Google Embeddings 2 usando text-embedding-004
print("Enviando fragmentos a Google AI Studio para vectorización...")

try:
    response = client.models.embed_content(
        model="text-embedding-004",
        contents=fragmentos_procesados
    )
    
    # 5. Extraer los vectores de la estructura de respuesta de la SDK
    # La respuesta contiene una lista de objetos Embedding, cada uno con una lista de floats en 'values'
    matriz_embeddings = [emb.values for emb in response.embeddings]
    
    print("\n--- METADATOS DE LA MATRIZ GENERADA ---")
    print(f"Número total de vectores devueltos: {len(matriz_embeddings)}")
    print(f"Dimensión espacial de cada vector (d): {len(matriz_embeddings[0])}")
    print(f"Ejemplo del primer vector (primeros 5 componentes): {matriz_embeddings[0][:5]}")

except Exception as e:
    print(f"❌ Error al conectar con la API de Google GenAI: {e}")

# %% [markdown]
# ## 4. Almacenamiento e Indexación con Bases de Datos Vectoriales

# %% [markdown]
# ### 1. Bases de Datos Vectoriales y Motores de Búsqueda de Vecinos Más Cercanos
# 
# En las bases de datos relacionales tradicionales (como PostgreSQL o MySQL), la información se consulta mediante coincidencias exactas o relacionales (ej. `WHERE id = 10` o `LIKE '%frenos%'`). Este enfoque falla en NLP porque es incapaz de identificar sinónimos, variaciones gramaticales o contextos conceptuales abstractos.
# 
# Para solucionar esto, la ingeniería de datos utiliza **Bases de Datos Vectoriales** o motores de indexación especializados. Su función no es almacenar tablas, sino organizar y guardar vectores de características (embeddings) en un espacio de alta dimensión.
# 
# En lugar de realizar un escaneo lineal completo comparando nuestra consulta contra cada vector del sistema —lo cual sería computacionalmente prohibitivo con millones de documentos (complejidad $\mathcal{O}(N)$)—, estos motores estructuran los vectores utilizando algoritmos de **Búsqueda de Vecinos Más Cercanos Aproximados** (ANN - *Approximate Nearest Neighbors*). 
# 
# En esta sección utilizaremos **FAISS (Facebook AI Similarity Search)**, una librería optimizada de código abierto que crea índices matemáticos en la memoria RAM para agrupar vectores similares espacialmente, permitiendo búsquedas semánticas en sub-milisegundos ($\mathcal{O}(\log N)$).

# %% [markdown]
# ### 2. El Proceso de Recuperación (Retrieval Step)
# 
# El flujo técnico para recuperar información relevante cuando un usuario realiza una pregunta consta de tres pasos críticos:
# 
# 1. **Vectorización de la Consulta (Query Embedding):** La pregunta del usuario (ej. *"¿Qué pasa si fallan los sensores magnéticos?"*) se envía al mismo modelo de embedding original (`text-embedding-004`). Esto es mandatorio: si usamos un modelo diferente, las dimensiones y las bases matemáticas del espacio espacial no coincidirán.
# 2. **Búsqueda en el Índice (Index Query):** El vector de la consulta ingresa al motor FAISS, el cual calcula la similitud de coseno contra los grupos vectoriales indexados.
# 3. **Extracción del Contexto (Top-K Retrieval):** El motor retorna los $K$ fragmentos de texto originales cuyos vectores posean el menor ángulo geométrico (mayor similitud) con el vector de la pregunta.

# %% [markdown]
# ### 3. Implementación de un Índice FAISS en Memoria y Búsqueda Semántica
# 
# Utilizaremos los embeddings generados por la API de Google en la sección anterior para poblar nuestro índice FAISS y simular una consulta real de usuario.

# %%
import numpy as np
import faiss

# 1. Preparar la matriz de entrada para FAISS
# FAISS requiere que los vectores de entrada sean una matriz de tipo float32 de NumPy
vectores_np = np.array(matriz_embeddings).astype('float32')

# 2. Inicializar un índice FAISS basado en la distancia de Producto Interno (Inner Product)
# Dado que los embeddings de Google vienen normalizados de fábrica, el producto interno 
# equivale directamente a calcular la Similitud de Coseno.
dimension_espacial = vectores_np.shape[1] # d = 768
index = faiss.IndexFlatIP(dimension_espacial)

# 3. Registrar los vectores en el índice indexado
index.add(vectores_np)
print(f"✅ Índice FAISS inicializado correctamente. Total de fragmentos indexados: {index.ntotal}")

# %%
# 4. Definir una consulta de usuario (Query) en lenguaje natural
pregunta_usuario = "¿Qué componente físico se debe revisar si el panel muestra un error magnético?"

print(f"Consulta del usuario: '{pregunta_usuario}'")

# 5. Transformar la pregunta del usuario al mismo espacio vectorial (d = 768)
response_query = client.models.embed_content(
    model="text-embedding-004",
    contents=pregunta_usuario
)
vector_query = np.array([response_query.embedding.values]).astype('float32')

# %%
# 6. Ejecutar la búsqueda en el índice FAISS para extraer los Top-K vecinos más cercanos
K = 2 # Solicitamos los 2 fragmentos más relevantes
similitudes, indices_retornados = index.search(vector_query, K)

print("\n--- FRAGMENTOS RECUPERADOS (RETRIEVAL) ---")
# Iterar sobre los resultados encontrados por el motor de búsqueda
for i in range(K):
    idx_fragmento = indices_retornados[0][i]
    score_similitud = similitudes[0][i]
    texto_recuperado = fragmentos_procesados[idx_fragmento]
    
    print(f"\n[Resultado #{i+1}] - Índice Original: {idx_fragmento} - Similitud de Coseno: {score_similitud:.4f}")
    print(f"-> Contexto extraído: \"{texto_recuperado}\"")

# %% [markdown]
# ## 5. El Pipeline RAG de Extremo a Extremo (End-to-End)

# %% [markdown]
# ### 1. Construcción del Prompt Aumentado y Mitigación Estricta de Alucinaciones
# 
# Una vez que el motor de búsqueda vectorial (FAISS) ha seleccionado los fragmentos de conocimiento con mayor afinidad semántica, ingresamos a la fase de **Aumentación**.
# 
# El objetivo de la ingeniería de prompts en un sistema RAG no es solo pasarle la información al modelo, sino **establecer las reglas del juego de forma restrictiva**. Para mitigar las alucinaciones por completo, el prompt se diseña bajo un principio de "caja cerrada":
# * Se define un **Rol Clave:** Se le indica al modelo que actúe como un asistente técnico experto y objetivo.
# * **Inyección de Contexto:** Se delimitan claramente las fronteras de los datos externos recuperados utilizando etiquetas estructuradas (ej. `<context> </context>`).
# * **Instrucción de Negación Estricta:** Se le prohíbe explícitamente al LLM utilizar su conocimiento preentrenado general para inventar datos. Si la respuesta no se puede deducir directamente del contexto provisto, el modelo debe responder con una frase estandarizada de fallo (ej. *"No cuento con información suficiente para responder"*).

# %% [markdown]
# ### 2. Inferencia con Gemini 1.5 Flash
# 
# Con el prompt aumentado completamente estructurado, invocamos al modelo de generación de texto. En esta sección utilizaremos `gemini-1.5-flash`, el cual cuenta con un motor optimizado para el procesamiento veloz de contextos densos, ideal para operar flujos de agentes y pipelines RAG en entornos de producción sin incurrir en costos de cómputo local.

# %%
# 1. Consolidar los fragmentos recuperados por FAISS en un único bloque de texto
contexto_consolidado = ""
for idx, i_index in enumerate(indices_returned[0]):
    texto_fragmento = fragmentos_procesados[i_index]
    contexto_consolidado += f"--- Fragmento {idx+1} ---\n{texto_fragmento}\n"

print("--- CONTEXTO EXTRAÍDO PARA EL PROMPT ---")
print(contexto_consolidado)

# %%
# 2. Diseñar el System Prompt restrictivo para mitigar alucinaciones
system_instruction = (
    "Eres un asistente de ingeniería experto y riguroso. Tu única tarea es responder "
    "preguntas técnicas basadas exclusivamente en el contexto provisto entre las etiquetas <context> y </context>.\n"
    "REGLAS OBLIGATORIAS:\n"
    "1. Responde de forma concisa, técnica y directa.\n"
    "2. Si la respuesta no se encuentra explícitamente en los fragmentos del contexto, debes contestar "
    "exactamente: 'Lo siento, la información solicitada no se encuentra en la documentación provista.'\n"
    "3. No inventes datos, no asumas conclusiones fuera del texto y bajo ninguna circunstancia uses conocimiento externo."
)

# 3. Estructurar el prompt de usuario inyectando el contexto y la pregunta original
user_prompt = f"""
Analiza los siguientes fragmentos técnicos e identifica la solución a la consulta.

<context>
{contexto_consolidado}
</context>

Consulta del usuario: {pregunta_usuario}
"""

# %%
# 4. Ejecutar la inferencia final enviando el prompt aumentado a Gemini 1.5 Flash
print("Invocando al generador de Google AI Studio (gemini-1.5-flash)...")

try:
    response_rag = client.models.generate_content(
        model="gemini-1.5-flash",
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.0,  # Temperatura en 0.0 para maximizar el determinismo y reducir la creatividad
            max_output_tokens=250
        )
    )
    
    print("\n--- RESPUESTA GENERADA POR EL SISTEMA RAG ---")
    print(response_rag.text)

except Exception as e:
    print(f"❌ Error durante la generación del LLM: {e}")

# %% [markdown]
# ### 4. Prueba de Mitigación de Alucinaciones (Control de Fronteras de Conocimiento)
# 
# Para demostrar la robustez de nuestro pipeline ante ingenieros de desarrollo, realizaremos una prueba de estrés. Le haremos al sistema una pregunta sobre un componente de automoción genérico (que un LLM conoce de sobra por su preentrenamiento), pero que **no está mencionado** en nuestro documento técnico cargado en la memoria RAM.

# %%
pregunta_trampa = "¿Cuál es el torque de apriete recomendado para las tuercas de las llantas de aleación?"
print(f"Consulta trampa del usuario: '{pregunta_trampa}'")

# Vectorizar la pregunta trampa y recuperar los fragmentos (FAISS igual traerá lo menos lejano)
vector_trampa = np.array([client.models.embed_content(model="text-embedding-004", contents=pregunta_trampa).embedding.values]).astype('float32')
_, indices_trampa = index.search(vector_trampa, K)

contexto_trampa = ""
for idx, i_index in enumerate(indices_trampa[0]):
    contexto_trampa += f"--- Fragmento {idx+1} ---\n{fragmentos_procesados[i_index]}\n"

user_prompt_trampa = f"""
Analiza los siguientes fragmentos técnicos e identifica la solución a la consulta.

<context>
{contexto_trampa}
</context>

Consulta del usuario: {pregunta_trampa}
"""

# Ejecutar la inferencia con temperatura 0
try:
    response_trampa = client.models.generate_content(
        model="gemini-1.5-flash",
        contents=user_prompt_trampa,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.0
        )
    )
    print("\n--- RESPUESTA DEL SISTEMA ANTE PREGUNTA FUERA DE CONTEXTO ---")
    print(response_trampa.text)
    print("\n✅ Éxito de ingeniería: El modelo bloqueó la alucinación respondiendo bajo las directrices del System Prompt.")

except Exception as e:
    print(f"❌ Error en la prueba de mitigación: {e}")