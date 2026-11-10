# Muchos usuarios, una GPU con vLLM

- **Por qué es relevante:** servir un modelo a una sola persona es fácil, porque cualquier servidor responde una petición tras otra. El problema aparece cuando llegan 50 a la vez. Una GPU que atiende a un usuario pasa casi todo el tiempo esperando, y la misma GPU puede escribir varias veces más tokens por segundo si atiende a muchos juntos. vLLM existe para aprovechar esa diferencia con tres ideas: continuous batching, PagedAttention (la KV-cache en bloques, como la memoria virtual de un sistema operativo) y prefix caching. Este repo las mide una por una con Gemma 4 en una GPU L4 de Cloud Run.
- **Para quién es:** desarrolladores que saben Python y HTTP, que ya llamaron alguna API de IA y quieren entender qué pasa cuando su app tiene muchos usuarios. No hace falta saber machine learning, JAX ni PyTorch.
- **Quién debería verlo:** quien tiene que decidir cuántos usuarios aguanta una GPU, por qué la latencia se dispara a partir de cierto punto, cuánto cuesta un millón de tokens con carga real o por qué un system prompt largo compartido sale más barato de lo que parece.
- **Serie completa:** unidad C3 de la Serie C, servir modelos abiertos con vLLM ([vLLM for Software Devs Field Guide](https://github.com/ykro/vllm-for-software-devs-field-guide)). Viene después de [C1, tu primer servidor vLLM con Gemma](https://github.com/ykro/first-vllm-server-with-gemma), que termina justo con la pregunta de esta unidad.

---

## Modelo mental

### 1. Una GPU con un solo usuario está casi ociosa

Generar texto tiene dos fases. El prefill lee todo el prompt de una vez, y el decode escribe la respuesta token por token. En el decode, cada paso tiene que leer todos los pesos del modelo desde la memoria de la GPU para producir un solo token. Con un usuario, ese viaje por la memoria se hace para un token. Con 32 usuarios en el mismo batch, el mismo viaje produce 32 tokens, uno para cada respuesta. Cada usuario va apenas un poco más lento y el servidor escribe muchas veces más tokens por segundo. Ese trade-off entre latencia y throughput es el tema completo de la unidad.

### 2. Continuous batching

Un batch clásico (static batching) junta peticiones, las procesa juntas y no acepta a nadie más hasta que termina la respuesta más larga. Una pregunta corta que llega a media corrida espera a que acabe el informe de 2,000 tokens que empezó antes. vLLM decide quién está en el batch en cada paso del decode. Una petición nueva entra en el siguiente paso, y la que termina sale en cuanto escribe su último token. El experimento 03 lo muestra con una línea de tiempo por petición, donde las respuestas cortas que llegan tarde terminan antes que las largas que empezaron primero.

### 3. PagedAttention, la KV-cache como memoria virtual

Para no releer todo en cada token, el modelo guarda lo que ya calculó de cada token anterior en la KV-cache, y esa memoria crece con cada token del contexto. Si el servidor reservara para cada petición un espacio contiguo del tamaño máximo posible (por ejemplo 8,192 tokens), la mayor parte de esa reserva quedaría vacía y cabrían muy pocos usuarios. PagedAttention hace lo mismo que un sistema operativo con la memoria virtual. Divide la KV-cache en bloques pequeños de tamaño fijo (16 tokens), le da a cada petición un bloque nuevo solo cuando lo necesita, de donde haya uno libre, y lleva una tabla que traduce la posición lógica de cada token al bloque físico donde vive. Casi no se desperdicia memoria y caben muchas más peticiones a la vez. Cuando los bloques se acaban, las peticiones nuevas esperan en la cola (`vllm:num_requests_waiting`), y si una petición activa necesita un bloque que no existe, vLLM saca a la más reciente (preemption) y la recalcula después. El experimento 04 lo ve en vivo desde `/metrics`.

### 4. Prefix caching

Casi todas las apps mandan el mismo system prompt largo (reglas, catálogo, tono) antes de una pregunta corta. Como la KV-cache ya está en bloques, vLLM puede reconocer que los primeros bloques de un prompt nuevo son idénticos a los de uno anterior y reutilizarlos. Solo la parte nueva paga prefill. El experimento 05 compara el tiempo al primer token con un prefijo compartido contra uno distinto por petición, y lee los contadores de aciertos del prefix cache.

---

## Resultados de verificación (`verify_many_users.py`)

> **Pendiente de la corrida en vivo.** Las cifras de esta sección salen de `make verify-live` contra vLLM `0.33.0` con `google/gemma-4-E4B-it` en una NVIDIA L4 de Cloud Run (`us-east4`). El juez guarda la corrida en `benchmarks/latest_results.json` y la salida de terminal en `benchmarks/terminal_output.txt`; esta tabla se llena con esa misma corrida.

| Experimento | Archivo | Qué mide | Resultado en L4 |
| :--- | :--- | :--- | :--- |
| **01. Uno por uno contra todos juntos** | `experiments/01_one_by_one_vs_together.py` | Tiempo total y tokens/s agregados de 8 peticiones en secuencia y en paralelo | Pendiente de la corrida en vivo |
| **02. Barrido de carga de 1 a 64 usuarios** | `experiments/02_load_sweep.py` | TTFT p50 y p95, latencia entre tokens, throughput, dólares por millón de tokens y la rodilla de la curva | Pendiente de la corrida en vivo |
| **03. Continuous batching** | `experiments/03_continuous_batching.py` | Línea de tiempo de 4 respuestas largas y 8 cortas que llegan después | Pendiente de la corrida en vivo |
| **04. KV-cache paginada bajo carga** | `experiments/04_paged_kv_cache.py` | Uso de la KV-cache, peticiones activas, en espera y preemptions según `/metrics` | Pendiente de la corrida en vivo |
| **05. Prefix caching** | `experiments/05_prefix_caching.py` | TTFT con system prompt compartido contra único y tasa de aciertos del prefix cache | Pendiente de la corrida en vivo |

### Cómo leer estas cifras

- **El juez revisa comportamiento.** Cada fila pasa si se cumple lo que la teoría predice: todos juntos dan más tokens/s que uno por uno, el TTFT p95 crece con la carga, cada respuesta corta termina antes que las largas que se enviaron antes, el uso de la KV-cache sube con la carga y un prefijo compartido arranca más rápido que uno único. Las cifras exactas cambian de una corrida a otra.
- **La rodilla.** El experimento 02 marca como rodilla el primer nivel de usuarios donde el tiempo que espera cada usuario por su respuesta completa (p50) crece en un factor mayor que el throughput. A partir de ahí, sumar usuarios agrega sobre todo espera.
- **El túnel cuenta.** Todos los tiempos incluyen el viaje de ida y vuelta por `gcloud run services proxy`, así que el TTFT medido es mayor que el que reporta vLLM por dentro.
- **La KV-cache de la L4 es grande.** Si el servidor publica su capacidad (`vllm:cache_config_info`), el experimento 04 ajusta el largo de las respuestas para que el nivel más alto pida un 25 % más de KV-cache de la que existe, con un tope de 768 tokens por respuesta. Si aun así no ves peticiones en espera, despliega con `VLLM_GPU_MEMORY_UTILIZATION=0.6` para dejarle menos memoria a la KV-cache a propósito.
- **El modo `--mock` no mide nada real.** El servidor falso de `vllm_lab/mock_server.py` imita el scheduler de vLLM (pasos de decode que se alargan con cada petición activa, una KV-cache de solo 4,096 tokens en bloques de 16, preemption y prefix caching) para que `make all` revise la lógica sin GPU. Sus tiempos son inventados.

---

## Estructura del repositorio

```text
.
├── benchmarks/
│   ├── pricing.json                         # Precio por hora de la instancia L4 (revísalo antes de decidir)
│   ├── latest_results.json                  # Lo escribe make verify-live
│   ├── terminal_output.txt                  # Salida íntegra del juez en vivo
│   └── timeline_continuous_batching.json    # Lo escribe el experimento 03 en vivo
├── experiments/
│   ├── 01_one_by_one_vs_together.py         # Secuencia contra paralelo: un usuario deja la GPU ociosa
│   ├── 02_load_sweep.py                     # De 1 a 64 usuarios: latencia contra throughput y la rodilla
│   ├── 03_continuous_batching.py            # Las respuestas cortas no esperan a las largas
│   ├── 04_paged_kv_cache.py                 # /metrics en vivo: la KV-cache se llena y aparece la cola
│   └── 05_prefix_caching.py                 # Un system prompt compartido se lee una sola vez
├── gcp/
│   ├── deploy_vllm.sh                       # vLLM + Gemma en Cloud Run con L4, concurrency 128
│   ├── stage_weights.sh                     # Copia los pesos a Cloud Storage una sola vez
│   └── teardown.sh                          # Borra el servicio
├── tests/
│   └── test_many_users.py                   # Suite pytest contra el servidor falso (9 tests)
├── vllm_lab/
│   ├── client.py                            # Clientes OpenAI sync y async apuntando a vLLM
│   ├── load.py                              # Generador de carga con asyncio y percentiles
│   ├── metrics.py                           # Lector de /metrics tolerante a nombres que cambian
│   ├── mock_server.py                       # vLLM falso con scheduler, KV-cache en bloques y prefix cache
│   └── cli.py                               # --mock y --base-url para todos los experimentos
├── Makefile
├── pyproject.toml
└── verify_many_users.py                     # El Outside Judge
```

---

## Paso a paso

### 1. Instalación

```bash
git clone https://github.com/ykro/many-users-one-gpu-with-vllm.git
cd many-users-one-gpu-with-vllm
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
```

La única dependencia es el SDK de OpenAI. vLLM corre en la GPU de Cloud Run, nunca en tu computadora.

### 2. Sin GPU, con el servidor falso

```bash
# ruff + pytest + el juez contra el servidor falso (menos de un minuto)
make all
```

Cada experimento también corre solo con `--mock`:

```bash
python experiments/02_load_sweep.py --mock
python experiments/04_paged_kv_cache.py --mock --levels 1,8,32,64
```

### 3. Con la GPU de verdad

vLLM necesita una GPU NVIDIA. El servicio usa una L4 on-demand en Cloud Run (`min-instances 0`), así que solo se cobra mientras una petición la mantiene despierta.

```bash
export GCP_PROJECT_ID=<tu-proyecto>

# Una vez: copia los pesos de Gemma a un bucket de la misma región (arranques en frío más rápidos)
bash gcp/stage_weights.sh

# Despliega vLLM 0.33.0 con Gemma 4 E4B, una sola instancia y concurrency 128
make deploy

# En otra terminal: el túnel autenticado hacia el servicio
gcloud run services proxy vllm-gemma --region us-east4 --port 8000

# El juez en vivo: escribe benchmarks/latest_results.json y benchmarks/terminal_output.txt
make verify-live
```

La corrida en vivo toma unos minutos. El primer request después de un rato sin uso espera el arranque en frío de la GPU. Si ya desplegaste el servicio en C1, `make deploy` lo actualiza con los flags de esta unidad (`--max-num-seqs`, `--enable-prefix-caching` y la concurrency de Cloud Run).

Para ver un experimento con detalle mientras el túnel está abierto:

```bash
python experiments/03_continuous_batching.py        # guarda benchmarks/timeline_continuous_batching.json
python experiments/04_paged_kv_cache.py --levels 1,16,64
make report                                         # el juez imprime además el resultado crudo de cada experimento
```

### 4. Al terminar

```bash
make teardown
```

Con `min-instances 0` un servicio sin tráfico no cobra GPU, pero borrarlo es la única forma de estar seguro.
