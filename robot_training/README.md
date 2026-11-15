# Robot training - policies para el ElRobot (NormaCore) con LeRobot

Todo lo necesario para pasar de **un dataset grabado con el ElRobot en NormaCore Station** a
**una policy entrenada, subida a Hugging Face y ejecutándose en el brazo**. Hay tres policies
(ACT, Diffusion Policy y SmolVLA) y, opcionalmente, las yemas de los dedos del paciente como entrada.

```
                              ┌─► convert_normacore.py ─► LeRobotDataset ─► train_act.sh / train_diffusion.sh ─┐
dataset NormaCore (Parquet) ──┤        [--hand dataset: + yemas de los dedos en el estado]                     ├─► inference/
   inspect_normacore.py       └─► normalize_cameras.py ─► un Parquet ─────► train_smolvla.sh ──────────────────┘   (local o remoto)
```

**Si eres un agente o llegas nuevo: lee primero la sección [A. Estado actual](#a-estado-actual-léelo-primero).**
Resume qué datos y modelos existen, qué está probado y qué falta.

**Por qué hace falta esto:** el ElRobot tiene 7 joints + gripper (8 servos Feetech ST3215, IDs 1..8).
LeRobot solo trae robots tipo SO-101 (6 motores) y Station graba en su propio esquema Parquet,
que no es un LeRobotDataset. Aquí está el conversor y los scripts de inferencia que hablan con
Station directamente.

| Fichero | Para qué |
|---|---|
| `setup_env.sh` | Crea el entorno: Python 3.12, LeRobot, norma-core y protobufs |
| `inspect_normacore.py` | Revisa un dataset antes de convertirlo (fps, huecos, cámaras) |
| `convert_normacore.py` | Parquet NormaCore → LeRobotDataset |
| `train_act.sh` | Entrena ACT (y sube el resultado si se le da un repo) |
| `train_diffusion.sh` | Entrena Diffusion Policy (sección 3c) |
| `hand_features.py` + `inference/hand_source.py` | Yemas de los dedos como entrada de la policy (sección 3d) |
| `inference/check_hand.py` | Comprueba que la fuente de yemas de inferencia coincide con el dataset |
| `normalize_cameras.py` + `train_smolvla.sh` | Fine-tune de SmolVLA con la receta de NormaCore (sección 3b) |
| `upload_model.py` | Sube un checkpoint a Hugging Face con su model card |
| `inference/run_local.py` | Inferencia con la policy en la misma máquina que el robot |
| `inference/policy_server.py` + `robot_client.py` | Inferencia con la policy en un servidor GPU / nube |
| `inference/selftest.py` | Prueba la inferencia sin robot, reproduciendo un episodio grabado |
| `inference/make_camera_refs.py` | Genera el banco para reconocer las cámaras en inferencia |

---

## A. Estado actual (léelo primero)

Actualizado el 15 de noviembre de 2026. La tarea es una sola: **"move device from right index to
right middle finger"** (el brazo lleva una anilla del dedo índice al corazón de una mano derecha).

### A.1 Datos

| Qué | Dónde | Notas |
|---|---|---|
| Dataset grabado | HF dataset `JustoDario/flexnsense_skill1` | 83 episodios, 15 766 frames, 10 Hz, 3 cámaras 224x224, 8 joints |
| Columnas de la mano | en ese mismo dataset (`hand_*`) | en los 83 episodios; mano detectada en el 99.5 % de los frames |
| LeRobotDataset, 43 ep | local: `JustoDario/flexnsense_skill1_lerobot` | estado de 8; el del primer ACT |
| LeRobotDataset, 63 ep | local: `JustoDario/flexnsense_skill1_63ep_lerobot` | estado de 8 |
| LeRobotDataset, 83 ep + yemas | local: `JustoDario/flexnsense_skill1_83ep_hand_lerobot` | estado de 13 (`--hand dataset`) |
| Parquet canónico, 43 ep | local: `flexnsense_skill1_canon.parquet` | entrada de SmolVLA |

Los LeRobotDataset no están en el Hub: se regeneran con `convert_normacore.py` en ~3 min (sección 1.3).

Cosas del dataset que rompen el entrenamiento si se ignoran:

- **El orden de `images[]` cambia entre sesiones** (Station no etiqueta las cámaras):
  episodios 1-20 `(side, gripper, table)`, 21-63 `(table, side, gripper)`, 64-83 `(table, gripper, side)`.
  Nunca uses `images[i]` por índice: `convert_normacore.py`, `normalize_cameras.py` e `inference/`
  reconocen las vistas por contenido. `hand_camera` es el índice de la vista `side`.
- **10 Hz con huecos de grabación** (61 en total): se colapsan, los frames quedan consecutivos.
- **Tres sesiones con la mano en sitios distintos** (misma calibración de servos): episodios 1-43,
  44-63 y 64-83. La cámara `table` cambió de encuadre a partir del 44. Un modelo entrenado solo con
  la primera sesión falla en las otras (ACT de 63 episodios sobre el episodio 70, que no vio:
  error mediano de 100 a 330 ticks en bucle abierto).
- En `selftest.py`, el ACT con yemas da en el episodio 70 un error mediano de 5 a 31 ticks por
  motor. No es comparable con el dato anterior: ese episodio sí está en su entrenamiento.
- **Los motores 5 y 7 casi no se mueven** (std 0.012 y 0.003 en `position_norm`).
- **El gripper se cierra porque el goal va por delante de la posición** (goal 0.025 frente a 0.037
  cerrado): una policy que promedia aprieta menos que las demos.
- Las columnas `hand_*` parecen calculadas con **WiLoR** (muñeca fija en `hand_keypoints_3d`,
  99.5 % de detección); MediaPipe sobre los mismos JPEG da ~0.01 de diferencia y detecta el 92.6 %.
  **El código que las genera no está en ninguna rama del remoto** (la rama `data_collection` solo
  tiene el `record_dataset.py` original): no se ha podido verificar su configuración.

### A.2 Modelos entrenados

| Modelo | Datos | Pasos | Dónde está | Probado en el brazo |
|---|---|---|---|---|
| ACT, chunk 30 | 43 ep, estado 8 | 30 000 | HF `JustoDario/flexnsense_act_skill1_rightindex2middle` | **Sí: regular.** No cerraba bien el gripper y el brazo temblaba |
| SmolVLA (NormaCore), chunk 50 | 43 ep, estado 8 | 12 000 | HF `JustoDario/flexnsense_smolvla` (público) | No |
| Diffusion Policy, horizon 16 | 43 ep, estado 8 | 20 000 de 30 000 (pausado) | solo local | No |
| ACT, chunk 30 | 63 ep, estado 8 | 30 000 | HF `JustoDario/flexnsense_act_skill1_rightindex2middle_63ep` (privado) | No |
| ACT, chunk 30, **con yemas** | 83 ep, estado 13 | 30 000 | HF `JustoDario/flexnsense_act_skill1_rightindex2middle_83ep_hand` (privado) | No |

En la máquina de entrenamiento (RTX PRO 4000 Blackwell, 24 GB) todo cuelga de `/opt/data/justo`:
`env.sh` (entorno de LeRobot), `checkpoints/<RUN>/`, `logs/<RUN>.log`, `datasets/flexnsense_skill1_raw`
(el dataset descargado) y `src/norma-core` (con el venv de SmolVLA en `software/ai/smolvla_py/.venv`).
Un entrenamiento ha terminado bien si su log acaba en `train exit 0`.

Tiempos medidos ahí (dependen de los pasos, no del número de episodios):

| Policy | Velocidad | Entrenamiento | VRAM | Inferencia por chunk (GPU) |
|---|---|---|---|---|
| ACT | 8.2 pasos/s, batch 32 | 30 000 pasos ≈ 1 h | 3 GB | ~20 ms |
| Diffusion | 2.7 pasos/s, batch 64 | 30 000 pasos ≈ 3 h | 11 GB | ~240 ms |
| SmolVLA | 0.54 pasos/s, batch 64 | 12 000 pasos ≈ 6 h | 13.5 GB | ~105 ms (1.2 GB) |

### A.3 Qué checkpoint y con qué opciones

Los scripts de `inference/` reconocen solos el tipo de checkpoint; solo cambia lo que hay que tener instalado.

| Checkpoint | `--checkpoint` | Hace falta además | Re-planifica cada |
|---|---|---|---|
| ACT, estado 8 | carpeta `pretrained_model` (o la descargada de HF) | nada | 15 ticks (1.5 s) |
| ACT, estado 13 (yemas) | carpeta `pretrained_model` | WiLoR en la máquina del robot: `WILOR_ROOT`, `--hand-source wilor`; lo normal es el montaje en dos procesos (sección 3d) | 15 ticks |
| Diffusion | carpeta `pretrained_model` | `diffusers` (`uv pip install -e "lerobot[diffusion]"`) | 8 ticks (0.8 s) |
| SmolVLA | carpeta con `elrobot.json` | `smolvla_py` en el path (lo pone `setup_env.sh`) | 15 ticks; `--n-action-steps` hasta 50 |

Orden recomendado siempre: `selftest.py` → `--dry-run` → `--max-ticks 30` → sin límite (sección 4).

### A.4 Qué falta

0. Hay dos prompts de un solo uso para el agente del portátil del robot: `prompt.txt` (ACT de 63
   episodios) y `prompt2.txt` (ACT con yemas). Cada uno se borra de la rama al terminar su prueba.
1. **Probar en el brazo** SmolVLA, ACT de 63 episodios y ACT con yemas, y compararlos: 10 intentos
   por policy desde las mismas posiciones, contando éxitos. La loss y `selftest.py` no predicen
   el comportamiento real (son bucle abierto sobre episodios, la mayoría vistos al entrenar).
2. **Verificar WiLoR en inferencia**: `inference/check_hand.py` en el portátil, sobre un episodio
   antiguo (p.ej. el 10) y uno nuevo (el 70). `hand_source.py` no se ha ejecutado nunca con WiLoR.
3. **Subir a la rama `data_collection` el código que escribe las columnas `hand_*`**, para igualar
   su configuración de WiLoR con la de `hand_source.py` si `check_hand.py` no da OK.
4. **Regenerar `inference/camera_refs.npz` con los 83 episodios** (`make_camera_refs.py`): el banco
   antiguo solo conoce el encuadre de las primeras sesiones.
5. Ideas no implementadas: temporal ensembling para ACT (pedir chunk en cada tick y promediar,
   contra el temblor), yemas de los dedos en SmolVLA, y más demos con recuperaciones desde posiciones malas.
6. Descartado, con motivo: preentrenar un modelo base con `venayc/elrobot-pickplace` (solo 12
   episodios y 5099 frames, 2 cámaras en otra posición; "pick up the pen" son 2 episodios) y
   generar episodios en simulación (en norma-core solo hay el URDF, sin escena ni cámaras).

### A.5 Reglas del repo que afectan a esta carpeta

Ver [`AGENTS.md`](../AGENTS.md) en la raíz: una rama por tarea y PR a `main` (esta carpeta vive en
`feature/robot-training`), **sin atribución a herramientas de IA** en commits, PRs ni documentos,
y **sin rayas largas** (U+2014) en ningún fichero ni mensaje: lo comprueba el CI.

---

## 0. Entorno (una vez por máquina)

Requisitos: Linux, GPU NVIDIA para entrenar, `git`, `ffmpeg` y [`uv`](https://docs.astral.sh/uv/).

```bash
bash setup_env.sh ~/robot_env      # venv + LeRobot 0.7.1 (ACT, Diffusion, SmolVLA) + norma-core + protobufs
source ~/robot_env/env.sh          # en CADA terminal nueva
hf auth login                      # una vez; token con permiso de escritura si vas a subir modelos
```

`env.sh` activa el venv y define `NORMA_CORE`, `HF_HOME` y `TORCH_HOME`. Los LeRobotDataset
convertidos se guardan en `$HF_HOME/lerobot/<repo_id>`. Si tu home tiene cuota, instala el
entorno en un disco con espacio (el venv ocupa ~3 GB y las cachés crecen).

Hay dos entornos más, opcionales y separados de este:

- **SmolVLA, solo para entrenar:** `smolvla_py` trae su propio venv (`uv sync` en
  `$NORMA_CORE/software/ai/smolvla_py`, lo hace `train_smolvla.sh`) porque fija `transformers==5.3.0`.
  La inferencia de SmolVLA sí corre en el entorno de LeRobot.
- **WiLoR, solo para policies con yemas de los dedos:** tiene que estar instalado en el mismo
  Python que ejecuta `run_local.py` o `robot_client.py` (sección 3d).

> GPUs muy nuevas (p.ej. RTX PRO 4000 Blackwell, `sm_120`) necesitan un PyTorch compilado con
> CUDA ≥ 12.8. Comprueba `python -c "import torch; print(torch.cuda.is_available())"`.

---

## 1. Entrenar una policy con un dataset concreto

Ejemplo completo con el dataset `JustoDario/flexnsense_skill1` (43 episodios). Para otro
dataset cambia solo los nombres.

### 1.1 Descargar el dataset grabado con Station

```bash
hf download JustoDario/flexnsense_skill1 --repo-type dataset --local-dir ~/data/flexnsense_skill1_raw
```

Estructura esperada: `data/<task>/episode_0001.parquet ...`, una fila por frame con
`joints` (8 × `position_norm`, `goal_norm`, `range_min`, `range_max`, ...), `images` (3 JPEG 224×224) y `task`.

### 1.2 Inspeccionarlo (no te lo saltes)

```bash
python inspect_normacore.py --src ~/data/flexnsense_skill1_raw --sheet camaras.jpg
```

Fíjate en dos cosas:

1. **fps y huecos.** Station publica a 10 Hz (dt = 0.100 s). Si el fps medio sale más bajo es
   porque hay huecos de grabación; el conversor los colapsa (los frames quedan consecutivos).
2. **`camaras.jpg`** (filas = episodios, columnas = `images[0..2]`). **El orden de las cámaras no
   es estable**: Station no etiqueta las imágenes y el orden depende de cómo arrancan las cámaras
   USB. En `flexnsense_skill1` los episodios 1-20 son `(side, gripper, table)` y los 21-43
   `(table, side, gripper)`. Mira la **primera fila** y apunta qué vista es cada columna: eso es
   `--ref-views`.

Nombres de vista usados aquí: `side` (lateral del brazo), `gripper` (cámara del gripper),
`table` (a ras de mesa, mirando la mano).

### 1.3 Convertir a LeRobotDataset

```bash
python convert_normacore.py \
    --src ~/data/flexnsense_skill1_raw \
    --repo-id JustoDario/flexnsense_skill1_lerobot \
    --ref-views side,gripper,table        # vistas de images[0],[1],[2] en el episodio 1
```

- `observation.state` = `position_norm` (8) · `action` = `goal_norm` (8) · motor ID 1..8, el 8 es el gripper.
- Tres vídeos: `observation.images.side`, `.gripper`, `.table`.
- Cada episodio se reordena por contenido de imagen; si alguno es ambiguo, el script **aborta**
  en vez de mezclar cámaras. La salida lista el orden detectado en cada episodio: revísala.
- Tarda ~2 min para 43 episodios. Resultado en `$HF_HOME/lerobot/JustoDario/flexnsense_skill1_lerobot`.

Comprobación rápida:

```bash
python -c "
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset('JustoDario/flexnsense_skill1_lerobot')
print(ds.meta.total_episodes, 'episodios |', ds.meta.total_frames, 'frames |', ds.meta.fps, 'fps |', ds.meta.camera_keys)"
```

### 1.4 Entrenar

```bash
tmux new -s train        # para que siga aunque cierres la terminal (Ctrl+b d para salir)

DATASET=JustoDario/flexnsense_skill1_lerobot \
RUN=act_flexnsense_skill1_c30 \
bash train_act.sh
```

| Variable | Por defecto | Qué es |
|---|---|---|
| `DATASET` | - | repo_id del LeRobotDataset del paso 1.3 |
| `RUN` | - | nombre del run (ver [nombres](#2-nombres-de-las-policies-entrenadas)) |
| `STEPS` / `BATCH` | 30000 / 32 | pasos y batch size |
| `CHUNK` | 30 | acciones que predice ACT de golpe: 30 = 3 s a 10 Hz |
| `N_ACTION_STEPS` | 15 | cuántas ejecuta antes de re-planificar (solo inferencia) |
| `SAVE_FREQ` | 5000 | cada cuántos pasos guarda checkpoint |
| `HF_REPO` | vacío | si se define, sube el checkpoint final a ese repo al terminar |
| `OUT_ROOT` / `LOG_DIR` | `./outputs/train` / `./outputs/logs` | dónde se guarda todo |

- **Checkpoints:** `outputs/train/<RUN>/checkpoints/<step>/pretrained_model/`; `checkpoints/last` apunta al más reciente.
- **Progreso:** `tr '\r' '\n' < outputs/logs/<RUN>.log | grep 'INFO.*step:' | tail -3`
- **Reanudar** si se corta: `RUN=<RUN> RESUME=1 bash train_act.sh`
- Si `outputs/train/<RUN>` ya existe y no es `RESUME=1`, LeRobot da error: usa otro `RUN`.

Referencia medida (RTX PRO 4000 Blackwell, 24 GB): 8.5 pasos/s con batch 32, 3 GB de VRAM,
**30 000 pasos ≈ 1 hora**. Con 43 episodios (8465 frames) la loss baja de ~15 a ~0.1 en 10 000 pasos.

---

## 2. Nombres de las policies entrenadas

| Qué | Convención | Ejemplo |
|---|---|---|
| Dataset grabado (HF, dataset) | `<user>/flexnsense_skill<N>` | `JustoDario/flexnsense_skill1` |
| LeRobotDataset convertido | `<user>/flexnsense_skill<N>_lerobot` | `JustoDario/flexnsense_skill1_lerobot` |
| Run local (`RUN`) | `<policy>_flexnsense_skill<N>_c<chunk>` | `act_flexnsense_skill1_c30`, `smolvla_flexnsense_skill1_c50` |
| Modelo en HF (`HF_REPO`) | `<user>/flexnsense_<policy>_skill<N>_<tarea_corta>` | `JustoDario/flexnsense_act_skill1_rightindex2middle` |

Si cambias algo relevante (chunk, pasos, subconjunto de episodios), refléjalo en `RUN`
(`..._c50`, `..._60k`) para no pisar un run anterior.

---

## 3. Subir la policy a Hugging Face

Automático al acabar el entrenamiento:

```bash
DATASET=JustoDario/flexnsense_skill1_lerobot RUN=act_flexnsense_skill1_c30 \
HF_REPO=JustoDario/flexnsense_act_skill1_rightindex2middle \
TASK="move device from right index to right middle finger" \
bash train_act.sh
```

O a mano, cualquier checkpoint:

```bash
CK=outputs/train/act_flexnsense_skill1_c30/checkpoints/last/pretrained_model
python upload_model.py --checkpoint $CK --repo-id JustoDario/flexnsense_act_skill1_rightindex2middle --dry-run
python upload_model.py --checkpoint $CK --repo-id JustoDario/flexnsense_act_skill1_rightindex2middle \
    --task "move device from right index to right middle finger"
```

- Sube `config.json`, `model.safetensors` (~206 MB), los processors (normalización) y
  `train_config.json` a la **raíz** del repo, más una model card en `README.md` (`--no-card` para no tocarlo).
- El repo debe existir, o usa `--create` (lo crea **privado**; `--public` para público).
- **Si el repo es público** quedan públicos los pesos, las estadísticas del dataset y
  `train_config.json` (incluye rutas locales).
- Para usarlo en otra máquina:
  `hf download <user>/<repo> --local-dir ./act_ckpt` y pasa `--checkpoint ./act_ckpt` a los scripts de inferencia.

---

## 3b. SmolVLA con el fine-tune de NormaCore

Alternativa a ACT: el fine-tune estándar de NormaCore (`norma-core/software/ai/smolvla_py`), que
parte de `lerobot/smolvla_base` y lee los Parquet de Station directamente (no usa el
LeRobotDataset). Su `train.py` está escrito para el SO-101; aquí se ejecuta sin modificar y
`train_smolvla.py` solo cambia lo que trae fijo:

| `train.py` de NormaCore | Con el ElRobot | Cómo |
|---|---|---|
| 6 joints | 8 | se lee del Parquet |
| 2 cámaras, por índice de `images[]` | 3, en orden fijo | `normalize_cameras.py` reordena cada episodio |
| suelo de la std 1e-4 | 0.02 | `--std-floor`; los motores 5 y 7 casi no se mueven (std 0.012 y 0.003) |

```bash
# 1) Un solo Parquet con las cámaras en orden (side, gripper, table). ~20 s.
python normalize_cameras.py --src ~/data/flexnsense_skill1_raw \
    --output ~/data/flexnsense_skill1_canon.parquet --ref-views side,gripper,table

# 2) Entrenar (crea el venv de smolvla_py la primera vez: transformers==5.3.0, aparte del de LeRobot)
tmux new -s smolvla
PARQUET=~/data/flexnsense_skill1_canon.parquet RUN=smolvla_flexnsense_skill1_c50 bash train_smolvla.sh
```

| Variable | Por defecto | Qué es |
|---|---|---|
| `PARQUET` / `RUN` | - | Parquet del paso 1 y nombre del run |
| `STEPS` / `BATCH` | 12000 / 64 | la receta de NormaCore son 30000 pasos; ver tiempos abajo |
| `SAVE_EVERY` | 2000 | cada cuántos pasos guarda checkpoint |
| `STD_FLOOR` | 0.02 | `1e-4` = comportamiento original de NormaCore |
| `OUT_ROOT` / `LOG_DIR` | `./outputs/train` / `./outputs/logs` | dónde se guarda todo |

- **Checkpoints:** `outputs/train/<RUN>/step-NNNNNN/` y `final/`, cada uno con `config.json`,
  `model.safetensors` (~900 MB), `stats.safetensors` y `elrobot.json` (vistas, dimensiones y la
  frase del dataset; es lo que usa la inferencia para reconocer el checkpoint).
- **Progreso:** `grep '^step' outputs/logs/<RUN>.log | tail -3`
- **No se puede reanudar** (el `train.py` de NormaCore no tiene `--resume`). El LR sigue un coseno
  ajustado a `STEPS`: los checkpoints intermedios sirven, pero no han terminado de bajar el LR.
- Chunk de 50 acciones (5 s a 10 Hz), el del modelo base. Solo se entrena el action expert
  (100 M de 450 M parámetros).

Referencia medida (RTX PRO 4000 Blackwell, 24 GB): **0.54 pasos/s con batch 64, 13.5 GB de VRAM**,
GPU al 100 %. 12 000 pasos ≈ 6 h (91 épocas de 8465 frames); 30 000 pasos ≈ 15 h.

**Subir a Hugging Face:** `upload_model.py` es solo para checkpoints LeRobot. Para estos:
`hf upload <user>/flexnsense_smolvla_skill<N>_<tarea> outputs/train/<RUN>/final . --repo-type model --private`
(`elrobot.json` incluye rutas locales y `stats.safetensors` las estadísticas del dataset).

**Inferencia:** los mismos scripts de la sección 4, pasando la carpeta del checkpoint
(`.../final` o `.../step-NNNNNN`) como `--checkpoint`. Se ejecuta en el entorno de LeRobot
(`setup_env.sh` ya pone `smolvla_py` en el path; probado con transformers 5.22).
- Latencia por chunk: ~200 ms en GPU, ~1.8 s en CPU (4 hilos). Con CPU, mejor el modo remoto.
- Ejecuta 15 de las 50 acciones antes de re-planificar (`--n-action-steps` para cambiarlo).
- Usa la frase del dataset guardada en `elrobot.json`; `--task "..."` para otra.
- El `scripts/run_policy.py` de NormaCore no sirve aquí: lee las cámaras por índice, asume 6
  motores y falla al predecir por mezclar bf16 con float32 (aquí se usa autocast).

---

## 3c. Diffusion Policy

Misma entrada que ACT (el LeRobotDataset del paso 1.3) y mismo uso que `train_act.sh`. Sin
lenguaje ni modelo preentrenado grande: ResNet18 por cámara + U-Net, 293 M de parámetros.

```bash
tmux new -s diffusion
DATASET=JustoDario/flexnsense_skill1_lerobot RUN=diffusion_flexnsense_skill1_h16 bash train_diffusion.sh
```

| Variable | Por defecto | Qué es |
|---|---|---|
| `STEPS` / `BATCH` / `SAVE_FREQ` | 40000 / 64 / 10000 | pasos, batch y cada cuánto guarda |
| `HORIZON` | 16 | acciones que predice: 1.6 s a 10 Hz (múltiplo de 8) |
| `N_ACTION_STEPS` | 8 | cuántas ejecuta antes de re-planificar: 0.8 s |
| `N_OBS_STEPS` | 2 | observaciones que ve: el frame actual y el anterior |
| `CROP` | 200 | recorte aleatorio de las imágenes 224x224 al entrenar |
| `INFER_STEPS` | 10 | pasos de denoising (DDIM) en inferencia |
| `HF_REPO`, `TASK`, `RESUME`, `OUT_ROOT`, `LOG_DIR` | | igual que en `train_act.sh` |

- Los valores por defecto de LeRobot (horizon 64, n_action_steps 32) son para 30 Hz; a 10 Hz serían
  6.4 s y 3.2 s. Los de aquí son los del Diffusion Policy original.
- Checkpoints, progreso, reanudar y subida (`upload_model.py`): igual que ACT (secciones 1.4 y 3).
- Necesita `diffusers`: `setup_env.sh` ya lo instala; en un entorno anterior,
  `uv pip install -e "lerobot[diffusion]"` (también en la máquina que haga la inferencia).
- **Inferencia:** los scripts de la sección 4 con la carpeta `pretrained_model`. Como la policy
  necesita dos frames consecutivos, `control.py` le pasa también el del tick anterior (en modo
  remoto el cliente lo envía en la misma petición). Re-planifica cada 8 ticks.

Referencia medida (RTX PRO 4000 Blackwell): 2.7 pasos/s con batch 64, 11 GB de VRAM;
30 000 pasos ≈ 3 h (227 épocas de 8465 frames).

---

## 3d. Yemas de los dedos como entrada de la policy

El script de data collection guarda en cada frame la mano detectada en la cámara `side`
(columnas `hand_camera`, `hand_detected`, `hand_is_right`, `hand_keypoints_2d`,
`hand_fingertips_2d`, `hand_keypoints_3d`). Como la tarea es llevar la anilla del índice al
corazón, a la policy se le dan las dos yemas ya calculadas: 5 valores más en `observation.state`.

| `observation.state` | Qué es |
|---|---|
| 0-7 | `motor_1.pos` ... `motor_8.pos` (`position_norm`) |
| 8, 9 | `hand.index_x`, `hand.index_y`: yema del índice en la imagen `side`, fracción de ancho y alto (0..1) |
| 10, 11 | `hand.middle_x`, `hand.middle_y`: yema del corazón |
| 12 | `hand.detected`: 1 si hay mano en ese frame |

En los frames sin mano se repite la última detección del episodio con `hand.detected = 0`
(`hand_features.py`, la misma regla al convertir y en inferencia). La acción sigue siendo de 8.

**Convertir y entrenar** (ACT o Diffusion; el resto, igual que en las secciones 1.4 y 3c):

```bash
python convert_normacore.py --src ~/data/flexnsense_skill1_raw \
    --repo-id JustoDario/flexnsense_skill1_83ep_hand_lerobot --ref-views side,gripper,table --hand dataset

DATASET=JustoDario/flexnsense_skill1_83ep_hand_lerobot RUN=act_flexnsense_skill1_83ep_hand_c30 bash train_act.sh
```

| `--hand` | De dónde salen las yemas | En inferencia |
|---|---|---|
| `none` (por defecto) | no se usan: estado de 8 | nada que hacer |
| `dataset` | columnas `hand_*` del Parquet (WiLoR) | `--hand-source wilor` (por defecto) |
| `mediapipe` | se recalculan al convertir con MediaPipe (CPU; `uv pip install mediapipe`) | `--hand-source mediapipe` |

- Con `--hand dataset` el conversor comprueba que `hand_camera` es el `images[]` que él asigna a
  `side`; si no coincide, aborta (cámaras mal asignadas).
- En `flexnsense_skill1` (83 episodios, 15 766 frames) las columnas del dataset tienen mano en el
  99.5 % de los frames. MediaPipe sobre los mismos JPEG, el 92.6 %, y en algunos episodios menos
  de la mitad; además difiere de las columnas en ~0.01 (mediana; 2 px de 224) y hasta 0.04 (p95).
  Por eso la fuente recomendada es `dataset` + WiLoR.
- **La fuente de inferencia tiene que ser la del entrenamiento.** Una policy entrenada con las
  columnas de WiLoR y ejecutada con MediaPipe recibe coordenadas desplazadas y pierde la mano a menudo.
- SmolVLA (sección 3b) no usa estas columnas todavía.

**Inferencia.** Los scripts de la sección 4 detectan solos que el checkpoint espera 13 valores.
Las yemas se calculan en la máquina del robot (`control.py`), sobre el JPEG de `side`, cada vez
que se pide un chunk; en modo remoto viajan dentro del estado, así que el servidor no necesita WiLoR.

```bash
export WILOR_ROOT=/ruta/a/WiLoR          # checkout de WiLoR con sus pesos (o --wilor-root)
cd inference
# 1) ¿da WiLoR lo mismo que hay grabado? Debe acabar en "OK: es la misma fuente que el dataset"
python check_hand.py --parquet ~/data/flexnsense_skill1_raw/data/<task>/episode_0070.parquet --hand-source wilor
# 2) sin robot, con las yemas grabadas (--hand dataset, por defecto) o calculadas en vivo (--hand live)
python selftest.py --parquet <episode.parquet> local --checkpoint <pretrained_model>
python selftest.py --parquet <episode.parquet> --hand live --hand-source wilor local --checkpoint <pretrained_model>
# 3) en el brazo
python run_local.py --checkpoint <pretrained_model> --bus-serial <SERIAL_FOLLOWER> --dry-run
```

- `hand_source.py` usa `hand_tracking/wilor_source.py` de este repo (`WilorTracker`, modo rápido
  fp16): necesita el entorno de WiLoR (torch, ultralytics) en el Python que ejecuta `control.py`.
  En una RTX 3050: ~90 ms por imagen y ~1.7 GB de VRAM, además de la policy.
- **WiLoR y LeRobot en entornos distintos: dos procesos en la misma máquina.** `policy_server.py`
  en el entorno de LeRobot y `robot_client.py --server localhost:9000 --hand-source wilor` en el
  de WiLoR. El cliente solo necesita numpy, pillow, pyarrow y `NORMA_CORE` (station_py usa la
  librería estándar), no LeRobot. Probado en la máquina de entrenamiento con el cliente en un
  venv sin torch y las yemas calculadas en vivo con MediaPipe.
- En cada petición de chunk se imprime `[mano N ms: índice (x, y) corazón (x, y)]` o `SIN MANO`.
- El paso 1 es obligatorio la primera vez: `check_hand.py` no se ha podido ejecutar con WiLoR en
  la máquina de entrenamiento (no tiene WiLoR), solo con MediaPipe. Si no da OK, la configuración
  de WiLoR de `hand_source.py` no es la del script de data collection y hay que igualarla.

---

## 4. Inferencia en el brazo

El robot y Station están en el portátil. Dos formas de ejecutar la policy:

| Modo | La policy corre en | El portátil necesita |
|---|---|---|
| **Local** - `run_local.py` | el portátil (CPU o GPU) | torch + LeRobot 0.7.1 + norma-core |
| **Remoto** - `policy_server.py` + `robot_client.py` | un servidor GPU / nube | numpy, pillow + norma-core |

En los dos modos, una policy con yemas de los dedos (estado de 13) necesita además WiLoR en el
portátil: las yemas se calculan junto al robot y viajan dentro del estado (sección 3d).

```
                Station (portátil, TCP :8888)
   cámaras ─┐   ┌────────────────────────────┐
   buses  ──┴──►│ inference/normvla (10 Hz)  │──► run_local.py / robot_client.py
                │ commands  ◄── sync write ──│◄── goals (registro 0x2A del follower)
                └────────────────────────────┘
   modo remoto:  robot_client.py ──TCP (túnel SSH)──► policy_server.py (GPU)
```

### 4.0 Seguridad y privacidad: léelo antes de mover el brazo

Estos scripts **NO implementan la capa de seguridad del proyecto** descrita en el
[README principal](../README.md#safety). Mueven el brazo según lo que predice una red neuronal.

| Capa de seguridad del proyecto | Estado en `inference/` |
|---|---|
| Tope de torque/corriente escrito en los registros del servo | **No.** No se escribe ningún límite de torque. |
| Límites software de posición | Parcial: el goal se recorta al rango de calibración de cada joint (`range_min/max`), no a una zona segura para el paciente. |
| Watchdog de visión: ante detección perdida, **soltar y parar, nunca sujetar** | **No.** No hay lectura de fuerza. Si Station deja de dar frames el script se para, y el brazo **se queda sujetando** la última pose (lo contrario de "soltar"). |
| Velocidad máxima baja | Parcial: límite de cambio del goal por tick (`--max-step-ticks`), calibrado para reproducir las demostraciones, no para ser lento. |
| E-stop físico | Responsabilidad de quien opera. |

Por tanto: **no usar con la mano de un paciente**. Para pruebas, sin nadie al alcance del brazo o
con un objeto inerte, y con la alimentación a mano. Integrar la policy con el controlador de
admitancia y su capa de seguridad (`feature/admittance-control`) es trabajo pendiente.

**Privacidad.** `inspect_normacore.py` (`camera_sheet.jpg`) y los scripts de inferencia
(`camera_preview.jpg`) guardan en disco frames completos de las cámaras para comprobar a ojo el
orden. Están en `.gitignore`, pero son imágenes completas: no los generes con un paciente en
plano y bórralos después. `camera_refs.npz` contiene miniaturas 32x32 de las imágenes del dataset.

### 4.1 Antes de empezar

1. **Station corriendo** con el follower y las 3 cámaras. El **leader SIN torque**: con
   `st3215-bus: "auto"` Station solo publica `inference/normvla` si exactamente un bus tiene torque.
2. En el portátil, norma-core con los protobufs de Python generados (los del teleop) y
   `export NORMA_CORE=/ruta/a/norma-core` (o usa `setup_env.sh`).
3. **Banco de cámaras** (una vez por dataset/escena). No está en el repo porque se deriva de las imágenes del dataset:
   ```bash
   cd inference
   python make_camera_refs.py --src ~/data/flexnsense_skill1_raw --ref-views side,gripper,table
   ```
   Mismo `--ref-views` que en la conversión. Crea `inference/camera_refs.npz` (~5 MB); cópialo al portátil.
4. **Prueba sin robot** (reproduce un episodio grabado como si fuera Station):
   ```bash
   python selftest.py --parquet ~/data/flexnsense_skill1_raw/data/<task>/episode_0030.parquet \
       local --checkpoint <pretrained_model>
   ```

### 4.2 Modo local

```bash
cd inference
python run_local.py --checkpoint <pretrained_model> --bus-serial <SERIAL_FOLLOWER> --dry-run
python run_local.py --checkpoint <pretrained_model> --bus-serial <SERIAL_FOLLOWER>
```

### 4.3 Modo remoto

En el servidor (GPU):

```bash
cd inference
python policy_server.py --checkpoint <pretrained_model>      # escucha en 127.0.0.1:9000
```

En el portátil:

```bash
ssh -N -L 9000:localhost:9000 usuario@servidor                 # terminal 1: túnel
cd inference
python robot_client.py --server localhost:9000 --bus-serial <SERIAL_FOLLOWER> --dry-run   # terminal 2
python robot_client.py --server localhost:9000 --bus-serial <SERIAL_FOLLOWER>
```

Sin SSH: `policy_server.py --host 0.0.0.0 --token SECRETO` y `robot_client.py --server IP:9000 --token SECRETO`.
El tráfico va **sin cifrar**; mejor el túnel.

### 4.4 Qué hace y qué protecciones tiene

En cada frame nuevo de Station (10 Hz, el ritmo del dataset):

1. Lee 8 joints + 3 JPEG de `inference/normvla`.
2. **Reconoce las cámaras por contenido** (`camera_refs.npz`) y guarda `camera_preview.jpg`
   (side | gripper | table). **Míralo antes de confirmar.** Si no está seguro se para; se puede
   forzar con `--camera-order table,side,gripper` (orden de `images[0..2]` de Station).
3. Cuando se queda sin acciones pide un chunk a la policy y ejecuta las primeras `n_action_steps`.
4. Convierte `goal_norm` a ticks: `range_min + g · (range_max − range_min)`.
5. Protecciones (valores calibrados con `flexnsense_skill1`):
   - `--max-step-ticks 150` (gripper `--gripper-max-step-ticks 400`): máximo cambio del goal por tick.
   - `--max-delta-ticks 600`: si un joint del brazo pide un goal a más de eso de su posición
     real, ese tick no se envía; tras `--max-aborts 5` seguidos, se para. El gripper queda fuera:
     al apretar un objeto su goal va muy por delante de la posición (hasta ~1000 ticks en las demos).
6. Envía un sync write al registro GOAL_POSITION (`0x2A`) del bus del follower.

Arranque y parada:

- Al arrancar lee la posición actual (`st3215/inference`), escribe goal = posición y **después**
  activa el torque, para que el brazo no salte. `--no-enable-torque` si ya lo activaste tú.
- `--dry-run`: predice y muestra los goals pero no envía nada (el brazo se queda sujetando).
- Antes de mover pide confirmación `[s/N]` (`--yes` para saltarla). `Ctrl-C` para parar.
- Al salir **el follower sigue con torque** sujetando la pose. `--torque-off-on-exit` lo suelta (**el brazo cae**).

Orden recomendado la primera vez: `selftest.py` → `--dry-run` → `--max-ticks 30` → sin límite,
siempre con alguien listo para cortar la alimentación.

Otras opciones: `--station host[:8888]`, `--motor-ids 1,...,8`, `--n-action-steps N`, `--task "..."`
(solo para policies con lenguaje), `--device cpu|cuda`.

### 4.5 SmolVLA en el portátil, paso a paso

Para correr en local el SmolVLA de la sección 3b (`JustoDario/flexnsense_smolvla`) en el portátil
que tiene el robot. Necesita GPU NVIDIA con ≥ 2 GB de VRAM libres (usa 1.2 GB) o, en su defecto, el
modo remoto (4.3). Todo se hace en el portátil.

**1. Código** (una vez, y `git pull` cada vez que cambie la rama):

```bash
git clone https://github.com/anatissera/leukerbad-hackathon.git      # si aún no lo tienes
cd leukerbad-hackathon && git checkout feature/robot-training && git pull
cd robot_training
```

**2. Entorno** (una vez; sáltalo si ya tienes el de `setup_env.sh`, p.ej. de probar ACT en local):

```bash
bash setup_env.sh ~/robot_env        # venv + LeRobot (con transformers) + norma-core + smolvla_py
```

En cada terminal nueva:

```bash
source ~/robot_env/env.sh
python -c "import torch; print(torch.cuda.is_available())"     # debe decir True
```

Si dice `False`, la policy correrá en CPU (~2 s por chunk, el brazo irá a tirones): arregla el
driver NVIDIA / instala un torch con CUDA, o usa el modo remoto.

**3. Checkpoint** (una vez, ~1.2 GB):

```bash
hf auth login                                  # solo si el repo es privado
hf download JustoDario/flexnsense_smolvla --local-dir ~/smolvla_ckpt
ls ~/smolvla_ckpt      # config.json  elrobot.json  model.safetensors  stats.safetensors
```

La carpeta tiene que contener `elrobot.json`: es lo que hace que los scripts carguen SmolVLA en
vez de una policy LeRobot. Para probar otro checkpoint (`step-NNNNNN`), cópialo a otra carpeta y
pásala en `--checkpoint`.

**4. Banco de cámaras** (una vez por escena; sáltalo si ya tienes `inference/camera_refs.npz`):

```bash
hf download JustoDario/flexnsense_skill1 --repo-type dataset --local-dir ~/data/flexnsense_skill1_raw
cd inference
python make_camera_refs.py --src ~/data/flexnsense_skill1_raw --ref-views side,gripper,table
```

**5. Prueba sin robot** (comprueba entorno, checkpoint y latencia; la primera vez necesita
internet para bajar el tokenizer de SmolVLM2):

```bash
cd inference
python selftest.py --ticks 60 \
    --parquet ~/data/flexnsense_skill1_raw/data/move_device_from_right_index_to_right_middle_finger/episode_0030.parquet \
    local --checkpoint ~/smolvla_ckpt
```

Tiene que acabar con `SELFTEST local: 60 sync writes`. Mira los `(inferencia N ms)`: la primera es
lenta (calentamiento); las demás deberían quedar por debajo de ~500 ms. Si pasan de ~1 s, usa
`--n-action-steps 25` o el modo remoto.

**6. Station.** Arráncalo con el follower y las 3 cámaras, y el **leader sin torque** (4.1, punto 1).
Apunta el serial del bus del follower (web de Station, `:8889`).

**7. En el brazo**, en este orden y con la alimentación a mano (lee 4.0):

```bash
cd inference
python run_local.py --checkpoint ~/smolvla_ckpt --bus-serial <SERIAL_FOLLOWER> --dry-run       # no mueve nada
python run_local.py --checkpoint ~/smolvla_ckpt --bus-serial <SERIAL_FOLLOWER> --max-ticks 30  # 3 s
python run_local.py --checkpoint ~/smolvla_ckpt --bus-serial <SERIAL_FOLLOWER>                 # Ctrl-C para parar
```

Antes de confirmar `[s/N]` abre `camera_preview.jpg` y comprueba que es `side | gripper | table`.

**Ajustes si no va fino**

| Síntoma | Prueba |
|---|---|
| Tirón o pausa cada 1.5 s (al pedir chunk) | `--n-action-steps 25` (re-planifica cada 2.5 s; máximo 50) |
| Reacciona tarde a lo que ve | `--n-action-steps 10` |
| `CUDA out of memory` | cierra lo que use la GPU (navegador, Station viewer) o `--device cpu` para descartar |
| `SmolVLA necesita la instrucción` | el checkpoint no trae la frase: `--task "move device from right index to right middle finger"` |
| `ModuleNotFoundError: smolvla` | no has hecho `source ~/robot_env/env.sh`, o el entorno es anterior a `setup_env.sh` |
| Se para con las cámaras | `--camera-order` (4.4, punto 2) o regenera el banco (paso 4) |

---

## 5. Estado: qué está probado y qué no

El resumen de datos y modelos está en la sección [A](#a-estado-actual-léelo-primero). Aquí, el detalle de las pruebas.

**Probado en la máquina de entrenamiento**
- Conversión de `flexnsense_skill1` (83 episodios, 15 766 frames), con y sin `--hand dataset`:
  cámaras coherentes en todos los episodios y `hand_camera` igual a la vista `side` asignada.
- Entrenamiento de ACT, Diffusion Policy y SmolVLA, y carga de sus checkpoints para inferencia.
- Inferencia de punta a punta contra un Station **simulado** (`selftest.py`), en local y en remoto,
  con ACT, ACT con yemas (columnas grabadas), Diffusion (con el frame anterior) y SmolVLA.
- `check_hand.py` con MediaPipe (da AVISO contra las columnas del dataset, como se espera).

**Probado en el brazo real**
- El ACT de 43 episodios, en el portátil del robot: se mueve, pero no cierra bien el gripper y tiembla.

**No implementado**
- La capa de seguridad del proyecto (ver 4.0). No usar con un paciente.

**No probado**
- En el brazo: SmolVLA, Diffusion, ACT de 63 episodios y ACT con yemas.
- La fuente WiLoR de inferencia (`hand_source.py`, sección 3d): escrita contra
  `hand_tracking/wilor_source.py` pero sin ejecutar; en la máquina de entrenamiento no hay WiLoR.
- `--hand mediapipe` de punta a punta dentro del entorno de LeRobot (MediaPipe se probó en un venv aparte).
- El reconocimiento de cámaras con otra iluminación o con las cámaras movidas.
- La sección 4.5 en un portátil con RTX 3050: los comandos son los probados aquí, las latencias no.

## 6. Problemas conocidos

- **`frame con 8 joints / 2 imágenes; se esperaban 8/3`:** Station tiene una cámara sin abrir (pone
  una imagen por cámara activa). El script dice cuál falta y guarda lo recibido en
  `camera_received.jpg`. Comprueba las 3 cámaras en la web de Station (:8889), el cable/puerto USB,
  y reinicia Station tras reconectar. La policy necesita las 3 vistas: no puede correr con 2.
- **`inference/normvla` no da frames:** el follower no tiene torque, o el leader también lo tiene.
- **Station llena el disco:** graba siempre en `station_data/` (~2 GB/hora con 3 cámaras).
- **`station.yaml`:** `formats: []` funciona; una lista de strings da error de parseo.
- **Cámaras en Linux:** además del grupo `dialout`, hace falta una regla udev que dé acceso a
  `/dev/bus/usb` a las cámaras UVC (grupo `plugdev`).
- **`git sparse-checkout add` en norma-core** borra los protobufs generados si `target/gen_python`
  no está en el cone (`setup_env.sh` ya lo incluye).
- **`dataset-mp4` de NormaCore** falla con estos Parquet; para verlos, `inspect_normacore.py`.
- **Inferencia en CPU lenta:** limita los hilos con `POLICY_THREADS=4`.
