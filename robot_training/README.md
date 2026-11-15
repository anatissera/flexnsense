# Robot training - policies para el ElRobot (NormaCore) con LeRobot

Todo lo necesario para pasar de **un dataset grabado con el ElRobot en NormaCore Station** a
**una policy ACT entrenada, subida a Hugging Face y ejecutándose en el brazo**.

```
dataset NormaCore (Parquet)  ──►  LeRobotDataset  ──►  entrenar ACT  ──►  subir a HF  ──►  inferencia en el brazo
   inspect_normacore.py        convert_normacore.py     train_act.sh     upload_model.py     inference/
```

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
| `upload_model.py` | Sube un checkpoint a Hugging Face con su model card |
| `inference/run_local.py` | Inferencia con la policy en la misma máquina que el robot |
| `inference/policy_server.py` + `robot_client.py` | Inferencia con la policy en un servidor GPU / nube |
| `inference/selftest.py` | Prueba la inferencia sin robot, reproduciendo un episodio grabado |
| `inference/make_camera_refs.py` | Genera el banco para reconocer las cámaras en inferencia |

---

## 0. Entorno (una vez por máquina)

Requisitos: Linux, GPU NVIDIA para entrenar, `git`, `ffmpeg` y [`uv`](https://docs.astral.sh/uv/).

```bash
bash setup_env.sh ~/robot_env      # venv + LeRobot 0.7.1 + norma-core + protobufs
source ~/robot_env/env.sh          # en CADA terminal nueva
hf auth login                      # una vez; token con permiso de escritura si vas a subir modelos
```

`env.sh` activa el venv y define `NORMA_CORE`, `HF_HOME` y `TORCH_HOME`. Los LeRobotDataset
convertidos se guardan en `$HF_HOME/lerobot/<repo_id>`. Si tu home tiene cuota, instala el
entorno en un disco con espacio (el venv ocupa ~3 GB y las cachés crecen).

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
| Run local (`RUN`) | `<policy>_flexnsense_skill<N>_c<chunk>` | `act_flexnsense_skill1_c30` |
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

## 4. Inferencia en el brazo

El robot y Station están en el portátil. Dos formas de ejecutar la policy:

| Modo | La policy corre en | El portátil necesita |
|---|---|---|
| **Local** - `run_local.py` | el portátil (CPU o GPU) | torch + LeRobot 0.7.1 + norma-core |
| **Remoto** - `policy_server.py` + `robot_client.py` | un servidor GPU / nube | numpy, pillow + norma-core |

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

---

## 5. Estado: qué está probado y qué no

**Probado**
- Conversión de `flexnsense_skill1` (43 episodios, 8465 frames): valores idénticos al Parquet y
  cámaras coherentes en todos los episodios.
- Entrenamiento ACT con LeRobot 0.7.1 y carga del checkpoint para inferencia (GPU ~20 ms, CPU 30-45 ms).
- Reconocimiento de cámaras: 43/43 episodios dejando fuera el episodio evaluado.
- Inferencia local y remota de punta a punta contra un Station **simulado** (`selftest.py`).

**No implementado**
- La capa de seguridad del proyecto (ver 4.0). No usar con un paciente.

**No probado**
- **Ningún script de `inference/` se ha ejecutado contra el robot real**: el arranque de torque y
  el envío real de comandos están escritos según el código de norma-core, pero sin verificar en hardware.
- El reconocimiento de cámaras con otra iluminación o con las cámaras movidas.
- Otras policies. Para SmolVLA de LeRobot, `train_act.sh` sirve de plantilla cambiando
  `--policy.type`; el port `smolvla_py` de NormaCore está escrito para SO-101 (6 joints, 2 cámaras)
  y fija `transformers==5.3.0`, incompatible con este entorno.

## 6. Problemas conocidos

- **`inference/normvla` no da frames:** el follower no tiene torque, o el leader también lo tiene.
- **Station llena el disco:** graba siempre en `station_data/` (~2 GB/hora con 3 cámaras).
- **`station.yaml`:** `formats: []` funciona; una lista de strings da error de parseo.
- **Cámaras en Linux:** además del grupo `dialout`, hace falta una regla udev que dé acceso a
  `/dev/bus/usb` a las cámaras UVC (grupo `plugdev`).
- **`git sparse-checkout add` en norma-core** borra los protobufs generados si `target/gen_python`
  no está en el cone (`setup_env.sh` ya lo incluye).
- **`dataset-mp4` de NormaCore** falla con estos Parquet; para verlos, `inspect_normacore.py`.
- **Inferencia en CPU lenta:** limita los hilos con `POLICY_THREADS=4`.
