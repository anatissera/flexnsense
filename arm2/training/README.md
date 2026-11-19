# Robot training - policies para el ElRobot (NormaCore) con LeRobot

Todo lo necesario para pasar de **un dataset grabado con el ElRobot en NormaCore Station** a
**una policy entrenada, subida a Hugging Face y ejecutándose en el brazo**. Hay tres policies
(ACT, Diffusion Policy y SmolVLA) y, opcionalmente, las yemas de los dedos del paciente como entrada.

```
                              ┌─► convert_normacore.py ─► LeRobotDataset ─► train_act.sh / train_diffusion.sh ─┐
dataset NormaCore (Parquet) ──┤        [--hand dataset: + yemas de los dedos en el estado]                     ├─► inference/
   inspect_normacore.py       └─► normalize_cameras.py ─► un Parquet ─────► train_smolvla.sh ──────────────────┘   (local o remoto)
```

**Para lanzar el modelo que usamos, ve a [Modelo elegido: cómo lanzarlo](#modelo-elegido-cómo-lanzarlo).**

**Si eres un agente o llegas nuevo: lee después la sección [A. Estado actual](#a-estado-actual-léelo-primero).**
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
| `eval_holdout.py` | Error de cada checkpoint en episodios apartados del entrenamiento (sección 3e) |
| `normalize_cameras.py` + `train_smolvla.sh` | Fine-tune de SmolVLA con la receta de NormaCore (sección 3b) |
| `upload_model.py` | Sube un checkpoint a Hugging Face con su model card |
| `inference/run_local.py` | Inferencia con la policy en la misma máquina que el robot |
| `inference/policy_server.py` + `robot_client.py` | Inferencia con la policy en un servidor GPU / nube |
| `inference/selftest.py` | Prueba la inferencia sin robot, reproduciendo un episodio grabado |
| `inference/make_camera_refs.py` | Genera el banco para reconocer las cámaras en inferencia |

---

## Modelo elegido: cómo lanzarlo

**El modelo que usamos es el ACT multitarea, variante `marks`**: repo de Hugging Face
`JustoDario/flexnsense_act_multitask_fingertips_marks` (privado), checkpoint del paso 90 000
(revisión `825196ab`). Un solo modelo hace las tres tareas; la frase de `--task` elige entre qué
dos dedos se mueve la anilla. Elegido el 16 de noviembre de 2026 tras las pruebas en el brazo.

| Tarea | `--task` | `--n-action-steps` |
|---|---|---|
| Índice a corazón | `move device from right index to right middle finger` | 15 |
| Corazón a anular | `move device from right middle to right ring finger` | 15 |
| Anular a meñique | `move device from right ring to right pinky finger` | 9 |

`--n-action-steps` es cuántas acciones de cada chunk se ejecutan antes de pedir otro (15 = 1.5 s,
9 = 0.9 s). Los valores de la tabla son los que mejor han ido en las pruebas en el brazo.

**Terminal 1: Station** (en la carpeta donde está el binario `station` y su `station.yaml`):

```bash
./camera_setup.sh     # solo con Station parado; repetir si se desenchufan cámaras o se reinicia el PC
./station --web 127.0.0.1:8889 --tcp 127.0.0.1:8888 --max-queue-disk-size 3G
```

En `http://127.0.0.1:8889` tienen que verse el follower y las 3 cámaras. El leader, sin torque.

**Terminal 2: la policy** (en `arm2/training/inference` de este repo):

```bash
conda activate lerobot                    # el entorno con LeRobot y WiLoR
export NORMA_CORE=/ruta/a/norma-core
export WILOR_ROOT=/ruta/a/WiLoR           # checkout de WiLoR con sus pesos

# índice a corazón
TASK="move device from right index to right middle finger"
python -s run_local.py --checkpoint ~/ckpt/multi_marks --task "$TASK" --bus-serial 5B3D050113 --torque-off-on-exit --n-action-steps 15

# corazón a anular
TASK="move device from right middle to right ring finger"
python -s run_local.py --checkpoint ~/ckpt/multi_marks --task "$TASK" --bus-serial 5B3D050113 --torque-off-on-exit --n-action-steps 15

# anular a meñique
TASK="move device from right ring to right pinky finger"
python -s run_local.py --checkpoint ~/ckpt/multi_marks --task "$TASK" --bus-serial 5B3D050113 --torque-off-on-exit --n-action-steps 9
```

- `5B3D050113` es el bus del follower. `python -s` evita los paquetes de `~/.local`.
- El script pregunta `¿Empezar a mover el brazo? [s/N]`. Antes de contestar mira
  `camera_preview.jpg`: tiene que ser `side | gripper | table`. Las yemas se buscan en `side`.
- Al arrancar tiene que imprimir `dedo origen: ..., dedo destino: ...; yemas dibujadas en la
  imagen side`. Sin `--task`, o con una frase mal escrita, usa índice a corazón sin dar error.
- `Ctrl-C` para parar. Con `--torque-off-on-exit` el brazo se suelta al salir: **cae**, sujétalo.
- No hay capa de seguridad del proyecto (sección 4.0): no usar con la mano de un paciente.

### Desde el orquestador: lanzar la policy y, al pulsar el botón, parar y volver a home

La policy **no termina sola**: no sabe cuándo ha acabado la tarea. Quien la da por terminada es
una persona que mira el brazo y pulsa un botón, tanto si ha salido bien como si es mejor
rendirse. El orquestador hace dos llamadas, y las dos se ejecutan con el entorno de la Terminal 2
(`conda activate lerobot`, `NORMA_CORE`, `WILOR_ROOT`) y **con `arm2/training/inference` como
directorio de trabajo** (los scripts usan imports planos).

**1. Lanzar la policy** como proceso hijo, sin esperar a que termine:

```bash
python -s run_local.py --checkpoint ~/ckpt/multi_marks --task "$TASK" \
    --bus-serial 5B3D050113 --n-action-steps 15 --torque-off-on-exit --yes
```

- `--task` y `--n-action-steps`: los de la tabla de arriba, según la tarea.
- `--yes` es obligatorio desde un programa: sin él el script espera un `[s/N]` por stdin.
- Tarda unos segundos en cargar el modelo y WiLoR. El brazo empieza a moverse cuando imprime
  `ENVIANDO goals. Ctrl-C para parar.` en stdout.
- Si el proceso termina **por sí solo**, es un fallo, sea cual sea su código de salida (Station
  dejó de dar frames, falta una cámara, demasiados aborts seguidos). El motivo está en su stdout.
  En ese caso, con `--torque-off-on-exit` el brazo ya se ha soltado.

**2. Al pulsar el botón**, ejecutar y esperar a que termine:

```bash
python -s stop_and_home.py --bus-serial 5B3D050113
```

| Código de salida | Significado | Estado del brazo |
|---|---|---|
| 0 | policy parada y brazo en home | en home, **sin torque** |
| 1 | no llegó a home (motor bloqueado, Station caído) o la policy no se dejó parar | **con torque**, sujetando donde esté |

El mismo ejemplo en Python:

```python
import subprocess

INFERENCE = "arm2/training/inference"       # cwd de las dos llamadas
BUS = "5B3D050113"

policy = subprocess.Popen(
    ["python", "-s", "run_local.py", "--checkpoint", CKPT, "--task", TASK, "--bus-serial", BUS,
     "--n-action-steps", "15", "--torque-off-on-exit", "--yes"], cwd=INFERENCE)

wait_for_button()                            # la persona confirma: ha acabado, o nos rendimos

done = subprocess.run(["python", "-s", "stop_and_home.py", "--bus-serial", BUS], cwd=INFERENCE)
policy.wait()                                # ya está parada; esto solo recoge el proceso
at_home = done.returncode == 0
```

Qué hace `stop_and_home.py`, y qué tiene que saber el orquestador:

- **No hace falta parar la policy antes**: lo hace él. Le manda `SIGTERM`, que la hace salir
  **sujetando** el brazo aunque se lanzara con `--torque-off-on-exit`. Si en 3 s no ha salido,
  `SIGKILL`, que tampoco suelta el torque. Encuentra el proceso por el pidfile que la policy deja
  en `/tmp/run_local-<bus>.pid`.
- **Nunca parar la policy con `SIGINT` / `Ctrl-C`** si lleva `--torque-off-on-exit`: esa es la
  salida que suelta el torque, y el brazo cae. `SIGTERM` (`policy.terminate()`) sí es seguro.
- Si no hay ninguna policy corriendo (ya había muerto), va directo a home. Se puede pulsar el
  botón siempre.
- Después abre el gripper, lleva los motores 1-7 a home, cierra el gripper y suelta el torque.
  La pose está en `home_pose.json` (ticks leídos de la web de Station con el brazo en home).
  Tarda unos segundos, según lo lejos que esté el brazo.
- Al acabar con 0 el follower queda sin torque, así que Station puede dar frames a la policy
  de otro brazo (con `st3215-bus: auto` solo publica si exactamente un bus tiene torque).
- `gripper_open` en `home_pose.json` es el tick del motor 8 con el gripper abierto. Mientras sea
  `null` el gripper **no se abre** antes de mover el brazo: mídelo en la web de Station y escríbelo.
- El camino a home es recto en articulaciones, sin planificar colisiones: la mano fuera antes
  de pulsar.
- Con salida 1 el brazo sigue con torque donde se quedó: hay que mirarlo antes de volver a
  pulsar. Volver a ejecutar `stop_and_home.py` lo reintenta.

A mano: `python -s go_home.py --check` dice a cuánto está de home sin mover nada, y
`python -s go_home.py` va a home sin parar ninguna policy.

**Una vez por máquina** (detalle en la sección [4.6](#46-act-multitarea-en-el-portátil-paso-a-paso)):

```bash
hf download JustoDario/flexnsense_act_multitask_fingertips_marks --revision 825196ab --local-dir ~/ckpt/multi_marks
ls ~/ckpt/multi_marks/hand.json           # tiene que existir: es lo que marca la variante marks
hf download JustoDario/flexnsense_skill1 --repo-type dataset --local-dir ~/data/flexnsense_skill1_raw
python make_camera_refs.py --src ~/data/flexnsense_skill1_raw --ref-views side,gripper,table
```

`--revision` fija el checkpoint probado. El repo puede recibir pasos posteriores del mismo
entrenamiento (llega a 100 000); sin `--revision` se descarga el último, que no está probado en el brazo.

**Si el script muere con `sin frames nuevos en inference/normvla`:** un servo ha entrado en
protección por sobrecarga y Station deja de publicar frames. El script imprime el estado de cada
servo y los últimos ticks enviados antes de salir. Ver la protección de motor bloqueado en la
sección 4.4 y la sección 6.

---

## A. Estado actual (léelo primero)

Actualizado el 16 de noviembre de 2026. Hay tres tareas con la misma forma, sobre una mano derecha:
llevar una anilla **del índice al corazón**, **del corazón al anular** y **del anular al meñique**
("move device from right index to right middle finger", etc.). Los modelos actuales son dos ACT
multitarea que hacen las tres. El elegido es la variante `marks`: los comandos para lanzarlo están
en [Modelo elegido: cómo lanzarlo](#modelo-elegido-cómo-lanzarlo), y el montaje completo del
portátil en la sección [4.6](#46-act-multitarea-en-el-portátil-paso-a-paso).

### A.1 Datos

| Qué | Dónde | Notas |
|---|---|---|
| Dataset grabado | HF dataset `JustoDario/flexnsense_skill1` | 226 episodios en tres tareas (83 de índice a corazón, 83 de corazón a anular, 60 de anular a meñique), 10 Hz, 3 cámaras 224x224, 8 joints, 1.3 GB |
| Columnas de la mano | en ese mismo dataset (`hand_*`) | en todos los episodios; mano detectada en el 99 % de los frames, salvo en anular a meñique (74 %) |
| LeRobotDataset, 43 ep | local: `JustoDario/flexnsense_skill1_lerobot` | estado de 8; el del primer ACT |
| LeRobotDataset, 63 ep | local: `JustoDario/flexnsense_skill1_63ep_lerobot` | estado de 8 |
| LeRobotDataset, 83 ep + yemas | local: `JustoDario/flexnsense_skill1_83ep_hand_lerobot` | estado de 13 (`--hand dataset`) |
| LeRobotDataset multitarea, 226 ep + yemas | local: `JustoDario/flexnsense_multi3_hand_lerobot` | estado de 13 (yema origen y destino según la tarea); variante `state` |
| El mismo, con las yemas dibujadas | local: `JustoDario/flexnsense_multi3_handmarks_lerobot` | además dos anillos en la imagen `side` (`--hand-marks`); variante `marks` |
| Episodios de validación | 12 de los 226 (4 por tarea), índices 0-based `10, 31, 51, 72, 93, 114, 134, 155, 173, 188, 203, 218` | apartados del entrenamiento de los dos ACT multitarea (sección 3e) |
| Parquet canónico, 43 ep | local: `flexnsense_skill1_canon.parquet` | entrada de SmolVLA |

Los LeRobotDataset no están en el Hub: se regeneran con `convert_normacore.py` (sección 1.3; ~3 min
los de 83 episodios, ~10 min los de 226).

Los datos de 83 episodios que siguen (órdenes de cámara, sesiones) son de la primera tarea; las
otras dos carpetas tienen la misma estructura y `images[]` en el orden `(side, gripper, table)`
o `(table, side, gripper)` según el episodio.

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
| ACT multitarea, yemas origen/destino | 166 ep (índice a corazón y corazón a anular) | 40 000 | solo local: `act_flexnsense_multi_166ep_hand_c30` | No |
| **ACT multitarea, 3 tareas, variante `state`** | 214 ep (226 menos 12 de validación), estado 13 | 60 000 de 100 000 | HF `JustoDario/flexnsense_act_multitask_fingertips_state` (privado) | No |
| **ACT multitarea, 3 tareas, variante `marks`** | lo mismo, con las yemas dibujadas en la imagen `side` | 90 000 de 100 000 (revisión `825196ab`) | HF `JustoDario/flexnsense_act_multitask_fingertips_marks` (privado) | **Sí: es el modelo elegido** |

Los dos ACT multitarea son los modelos actuales (sección 4.6); el elegido es `marks`. Son la misma receta (chunk 30,
batch 32, mismas 214 demos) y solo cambian en cómo reciben las yemas del dedo origen y destino:
`state` las recibe como 5 números en el estado; `marks`, además, dibujadas en la imagen `side`.
El entrenamiento nocturno de los dos se cortó a mitad (los PCs del aula se reinician a las 06:00),
así que lo subido son checkpoints intermedios validados. El paso exacto de cada repo está en su
model card (`README.md` del repo). El de `marks` se reanudó y el repo va recibiendo pasos
posteriores: el probado en el brazo es el de 90 000 (revisión `825196ab`).

Error medio en los 12 episodios de validación (`eval_holdout.py`, ticks frente a los goals del
operador, bucle abierto; menor es mejor):

| Checkpoint | Media | índice a corazón | corazón a anular | anular a meñique |
|---|---|---|---|---|
| `state`, paso 40 000 | 53.6 | 49.6 | 53.8 | 57.3 |
| `state`, paso 60 000 (el subido) | 52.0 | 48.7 | 50.7 | 56.6 |
| `marks`, paso 40 000 | 54.1 | 49.0 | 55.6 | 57.7 |
| `marks`, paso 50 000 (el subido) | 52.9 | 47.4 | 54.3 | 56.9 |

Las dos variantes quedan prácticamente igual en esta medida: no dice cuál se comporta mejor en el
brazo, eso solo se sabe probándolas.

En la máquina de entrenamiento (RTX PRO 4000 Blackwell, 24 GB) todo cuelga de `/opt/data/justo`:
`env.sh` (entorno de LeRobot), `checkpoints/<RUN>/`, `logs/<RUN>.log`, `datasets/flexnsense_skill1_raw`
(el dataset descargado) y `src/norma-core` (con el venv de SmolVLA en `software/ai/smolvla_py/.venv`).
Un entrenamiento ha terminado bien si su log acaba en `train exit 0`. **Los PCs del aula se apagan
y reinician todos los días a las 06:00**: un entrenamiento largo tiene que acabar antes o
reanudarse después con `RESUME=1` (los checkpoints guardados no se pierden).

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
| ACT multitarea `state` / `marks` | carpeta descargada de HF | lo mismo que la fila anterior, más `--task "<frase de la tarea>"`, que elige los dedos (sección 4.6) | 15 ticks |
| Diffusion | carpeta `pretrained_model` | `diffusers` (`uv pip install -e "lerobot[diffusion]"`) | 8 ticks (0.8 s) |
| SmolVLA | carpeta con `elrobot.json` | `smolvla_py` en el path (lo pone `setup_env.sh`) | 15 ticks; `--n-action-steps` hasta 50 |

Orden recomendado siempre: `selftest.py` → `--dry-run` → `--max-ticks 30` → sin límite (sección 4).

### A.4 Qué falta

0. Hay prompts de un solo uso en la carpeta: `prompt.txt` (ACT de 63 episodios) y `prompt2.txt`
   (ACT con yemas) para el agente del portátil del robot, y `prompt3.txt`, el del entrenamiento
   nocturno del 16 de noviembre, que ya no aplica. Se borran de la rama cuando dejen de hacer falta.
1. **Medir el modelo elegido** (`marks`, ver "Modelo elegido"): 10 intentos por tarea desde las
   mismas posiciones, contando éxitos. Se eligió probándolo en el brazo, sin ese recuento. La
   loss, `eval_holdout.py` y `selftest.py` no predicen el comportamiento real (son bucle abierto).
2. Hecho: WiLoR en inferencia verificado con `inference/check_hand.py` en el portátil (episodio 32
   de corazón a anular: diferencia 0.0 con las columnas grabadas).
3. Hecho: el código que escribe las columnas `hand_*` está en la rama `data_collection`
   (`arm2/data_collection/hand_keypoints.py`), con PR abierto a `main`.
4. **Regenerar `inference/camera_refs.npz` en el portátil con los 226 episodios**
   (`make_camera_refs.py`, paso 4 de la sección 4.6): un banco antiguo solo conoce el encuadre de
   las primeras sesiones. No está en git, hay que generarlo en cada máquina.
5. **Terminar el ACT `marks` hasta 100 000 pasos** (reanudado el 16 de noviembre) y decidir si
   merece la pena terminar también el `state`, que entre 40 000 y 60 000 pasos casi no mejora.
6. Probar en el brazo el temporal ensembling (`--replan-every`, sección 3e). Ideas no
   implementadas: yemas de los dedos en SmolVLA y más demos con recuperaciones desde posiciones malas.
7. Descartado, con motivo: preentrenar un modelo base con `venayc/elrobot-pickplace` (solo 12
   episodios y 5099 frames, 2 cámaras en otra posición; "pick up the pen" son 2 episodios) y
   generar episodios en simulación (en norma-core solo hay el URDF, sin escena ni cámaras).

### A.5 Reglas del repo que afectan a esta carpeta

Ver [`docs/contributing.md`](../docs/contributing.md) en la raíz: una rama por tarea y PR a `main` (esta carpeta vive en
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

Sin GPU local, los notebooks de LeRobot entrenan en Colab. No son nuestro flujo (no leen
los datasets de Station sin convertirlos primero, sección 1.1), pero sirven para ver el
entrenamiento de principio a fin con una GPU prestada:

- [training-act.ipynb](https://colab.research.google.com/github/huggingface/notebooks/blob/main/lerobot/training-act.ipynb)
- [training-smolvla.ipynb](https://colab.research.google.com/github/huggingface/notebooks/blob/main/lerobot/training-smolvla.ipynb)

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

- `hand_source.py` usa `arm1/hand_tracking/wilor_source.py` de este repo (`WilorTracker`, modo rápido
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

## 3e. Un solo ACT para varias tareas dedo a dedo

El dataset tiene tres tareas con la misma forma: índice a corazón, corazón a anular y anular a
meñique. ACT no entiende lenguaje, pero no hace falta: las 5 entradas de la mano son siempre
"yema del dedo origen, yema del dedo destino, mano detectada", y los dos dedos salen de la frase
de la tarea de cada episodio (`hand_features.task_fingers`). Para la policy las tres tareas son
la misma, y los episodios de una ayudan a las otras.

- **Conversión:** `convert_normacore.py --hand dataset` sobre la carpeta con todas las tareas.
  No hay que indicar nada: cada episodio usa los dedos de su frase.
- **Inferencia:** hay que pasar la frase, que es la que elige los dedos:
  `--task "move device from right middle to right ring finger"`. Sin `--task` se usa índice a
  corazón, y por eso el modelo de 83 episodios sigue funcionando igual que antes.
- **Variante con marcas (`--hand-marks`):** además de los números, se dibujan dos anillos en la
  imagen `side`, verde en el origen y magenta en el destino. Hay que entrenar con `HAND_MARKS=1`
  para que cada checkpoint lleve `hand.json`; la inferencia lo lee y dibuja las mismas marcas
  sin ninguna opción extra.
- **Validación:** `EPISODES="[...]"` en `train_act.sh` entrena solo con esos episodios, y
  `eval_holdout.py` mide cada checkpoint en los que se apartaron (error medio en ticks frente a
  los goals del operador, por tarea). Sirve para ver si converge y elegir checkpoint.
- **Temporal ensembling:** es solo de inferencia. `--replan-every N` pide un chunk cada N ticks
  y promedia los chunks que se solapan (`--ensemble-coeff 0.01`, como el ACT original). Con
  yemas, cada petición ejecuta WiLoR (unos 90 ms), así que N = 1 no cabe en 100 ms en una RTX
  3050: empieza por `--replan-every 5`. No está probado en el brazo.

```bash
python convert_normacore.py --src ~/data/flexnsense_skill1 --repo-id <user>/<nombre>_lerobot --hand dataset [--hand-marks]
DATASET=<user>/<nombre>_lerobot RUN=<run> EPISODES="[0,1,2,...]" HAND_MARKS=1 STEPS=100000 SAVE_FREQ=10000 bash train_act.sh
python eval_holdout.py --src ~/data/flexnsense_skill1 --holdout holdout.json --run outputs/train/<run>
```

Avisos:

- En los episodios de anular a meñique la mano se detecta en torno al 74 % de los frames (99 %
  en las otras dos tareas); se pierde sobre todo a mitad del episodio. En esos frames la policy
  recibe la última posición conocida con `hand.detected = 0`, y las marcas se quedan donde estaban.
- No edites `train_act.sh` mientras hay un entrenamiento lanzado con él: bash lee el script
  según lo ejecuta y el final (la subida) falla. Los checkpoints no se pierden.
- **`hand.json` se escribe al final de `train_act.sh`.** Si un entrenamiento con `HAND_MARKS=1` se
  corta, sus checkpoints no lo tienen y se cargan como si fueran de la variante sin marcas: la
  inferencia y `eval_holdout.py` funcionan sin dar error, pero con la imagen sin anillos y peor
  resultado. Antes de validar o subir uno de esos checkpoints:
  `echo '{"marks": true}' > <checkpoint>/pretrained_model/hand.json`. Al reanudar con `RESUME=1`
  hay que volver a pasar `HAND_MARKS=1`.

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
   - Motor bloqueado (`--stall-lead-limit 55`, `--gripper-current-limit 100`): si un motor lleva
     0.8 s parado (menos de 15 ticks de avance) con el goal a más de 55 ticks, el goal se retiene
     a 45 ticks de la posición actual, siguiéndola si algo empuja el motor hacia atrás. Se libera
     cuando la policy pide ir hacia el otro lado o el motor avanza más de 30 ticks. Sin esto el
     servo entra en protección por sobrecarga (estado `0x20`) y Station deja de publicar frames.
     Le pasaba al gripper al abrir contra la anilla, que es elástica y lo empuja de vuelta.
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
git clone https://github.com/anatissera/flexnsense.git      # si aún no lo tienes
cd flexnsense && git pull
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

### 4.6 ACT multitarea en el portátil, paso a paso

Para correr en el portátil del robot los dos ACT multitarea de la sección A.2. Una sola policy
hace las tres tareas; la frase de `--task` le dice entre qué dos dedos.

| | Variante `state` | Variante `marks` |
|---|---|---|
| Repo de HF (privado) | `JustoDario/flexnsense_act_multitask_fingertips_state` | `JustoDario/flexnsense_act_multitask_fingertips_marks` |
| Cómo recibe los dedos | 5 números en el estado: yema origen (x, y), yema destino (x, y), mano detectada | lo mismo, y además un anillo verde en el origen y uno magenta en el destino dibujados en la imagen `side` |
| Fichero que la distingue | ninguno | `hand.json` en la carpeta del checkpoint; la inferencia lo lee sola |
| Cómo se lanza | igual | igual |

**Qué hace falta en el portátil**

- El código de esta rama y el entorno de `setup_env.sh` (pasos 1 y 2 de la sección 4.5).
- `hf auth login` con una cuenta que vea los repos: los dos modelos y el dataset son privados.
- **WiLoR** (un checkout con sus pesos) y GPU: las yemas se calculan en vivo sobre la cámara
  `side`, con el mismo modelo que anotó el dataset. Usa `arm1/hand_tracking/wilor_source.py` de este
  repo. Es lo único de esta lista que no se ha podido probar fuera del portátil.
- Station con el follower y las 3 cámaras, y el leader sin torque (4.1, punto 1).

**1. Checkpoints** (~210 MB cada uno):

```bash
hf download JustoDario/flexnsense_act_multitask_fingertips_state --local-dir ~/ckpt/multi_state
hf download JustoDario/flexnsense_act_multitask_fingertips_marks --local-dir ~/ckpt/multi_marks
ls ~/ckpt/multi_marks      # tiene que estar hand.json; en multi_state no existe
```

Repite la descarga de `marks` cuando el repo se actualice (su model card dice el paso).

**2. Dataset y banco de cámaras** (el banco no está en git; hazlo con el dataset completo):

```bash
hf download JustoDario/flexnsense_skill1 --repo-type dataset --local-dir ~/data/flexnsense_skill1_raw
cd inference
python make_camera_refs.py --src ~/data/flexnsense_skill1_raw --ref-views side,gripper,table
```

Debe listar los tres órdenes de `images[]` del dataset y guardar unas 4850 miniaturas por vista.

**3. ¿Da WiLoR lo mismo que el dataset?** Obligatorio la primera vez, en el entorno de WiLoR:

```bash
export WILOR_ROOT=/ruta/a/WiLoR
D=~/data/flexnsense_skill1_raw/data
python check_hand.py --hand-source wilor --parquet $D/move_device_from_right_middle_to_right_ring_finger/episode_0032.parquet
```

Tiene que acabar en `OK: es la misma fuente que el dataset`. Si da AVISO, la policy recibiría
coordenadas distintas de las del entrenamiento: no sigas hasta igualar la configuración de WiLoR
de `inference/hand_source.py` con la del script que grabó el dataset.

**4. Prueba sin robot**, con un episodio de validación de cada tarea (no vistos al entrenar:
`episode_0032` de las dos primeras carpetas, `episode_0023` de la del meñique). `selftest.py`
coge la frase del propio Parquet:

```bash
python selftest.py --ticks 60 --parquet $D/move_device_from_right_middle_to_right_ring_finger/episode_0032.parquet \
    local --checkpoint ~/ckpt/multi_marks
```

Debe imprimir `dedo origen: middle, dedo destino: ring` (con `marks`, seguido de `yemas dibujadas
en la imagen side`) y acabar en `SELFTEST local: 60 sync writes`. Esto usa las yemas grabadas;
añade `--hand live --hand-source wilor` antes de `local` para calcularlas con WiLoR.

**5. En el brazo.** Si WiLoR y LeRobot están en el mismo entorno de Python, un solo proceso:

```bash
TASK="move device from right middle to right ring finger"
python run_local.py --checkpoint ~/ckpt/multi_marks --task "$TASK" --bus-serial <SERIAL_FOLLOWER> --dry-run
python run_local.py --checkpoint ~/ckpt/multi_marks --task "$TASK" --bus-serial <SERIAL_FOLLOWER> --max-ticks 30
python run_local.py --checkpoint ~/ckpt/multi_marks --task "$TASK" --bus-serial <SERIAL_FOLLOWER>
```

Si están en entornos distintos (lo normal), dos procesos en el mismo portátil:

```bash
# terminal 1, entorno de LeRobot: la policy
source ~/robot_env/env.sh && cd inference
python policy_server.py --checkpoint ~/ckpt/multi_marks

# terminal 2, entorno de WiLoR (necesita numpy, pillow, pyarrow y NORMA_CORE; no LeRobot): el robot
export NORMA_CORE=~/robot_env/norma-core WILOR_ROOT=/ruta/a/WiLoR && cd inference
python robot_client.py --server localhost:9000 --hand-source wilor --task "$TASK" --bus-serial <SERIAL_FOLLOWER> --dry-run
```

Para cambiar de variante, cambia solo `--checkpoint` (y reinicia `policy_server.py`). Para cambiar
de tarea, solo `--task`.

**Qué tener en cuenta**

- **`--task` es obligatorio en la práctica.** Las frases son exactamente las del dataset:
  `move device from right index to right middle finger`,
  `move device from right middle to right ring finger` y
  `move device from right ring to right pinky finger`. Sin `--task`, o con una frase que no se
  reconoce, se usa índice a corazón **sin dar error**. Comprueba siempre la línea
  `dedo origen: ..., dedo destino: ...` que se imprime al arrancar.
- En cada petición de chunk se imprime `[mano N ms: <origen> (x, y) <destino> (x, y)]` o
  `SIN MANO`. Si sale `SIN MANO` a menudo, la policy trabaja con la última posición conocida y,
  en `marks`, los anillos se quedan donde estaban. En los datos de anular a meñique la mano se
  pierde en torno a un 26 % de los frames, sobre todo a mitad del episodio: es la tarea con más riesgo.
- Antes de confirmar `[s/N]` abre `camera_preview.jpg` y comprueba que es `side | gripper | table`.
  Las yemas se buscan en `side`: si las cámaras están mal asignadas, falla todo lo demás.
- WiLoR añade unos 90 ms a cada petición de chunk (RTX 3050), una cada 1.5 s. Para el temporal
  ensembling empieza por `--replan-every 5`, no por 1 (sección 3e).
- La policy y WiLoR comparten la GPU del portátil: WiLoR ocupa unos 1.7 GB, además de la policy.
- La validación deja las dos variantes empatadas (tabla de A.2). En el brazo se ha elegido
  `marks`: ver [Modelo elegido: cómo lanzarlo](#modelo-elegido-cómo-lanzarlo).
- Vale todo lo de la sección 4.0: no hay capa de seguridad, no usar con la mano de un paciente.

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
- Los dos ACT multitarea, descargados de Hugging Face como se hará en el portátil: `selftest.py`
  en local con un episodio de validación de cada tarea, y la variante `marks` también en dos
  procesos (`policy_server.py` + cliente). Con las yemas grabadas en el dataset, no con WiLoR en vivo.
- `make_camera_refs.py` con los 226 episodios (4850 miniaturas por vista, tres órdenes de `images[]`).

**Probado en el brazo real**
- El ACT de 43 episodios, en el portátil del robot: se mueve, pero no cierra bien el gripper y tiembla.
- El ACT multitarea `marks` (paso 90 000), en el portátil del robot (RTX 3050 Ti), con WiLoR en
  vivo en el mismo proceso que la policy: es el modelo elegido. Sin recuento de éxitos por tarea.
- `check_hand.py` con WiLoR en el portátil: diferencia 0.0 con las columnas `hand_*` del dataset.

**No implementado**
- La capa de seguridad del proyecto (ver 4.0). No usar con un paciente.

**No probado**
- En el brazo: SmolVLA, Diffusion y el ACT de 63 episodios.
- `--hand mediapipe` de punta a punta dentro del entorno de LeRobot (MediaPipe se probó en un venv aparte).
- El reconocimiento de cámaras con otra iluminación o con las cámaras movidas.
- La sección 4.5 en un portátil con RTX 3050: los comandos son los probados aquí, las latencias no.

## 6. Problemas conocidos

- **`frame con 8 joints / 2 imágenes; se esperaban 8/3`:** Station tiene una cámara sin abrir (pone
  una imagen por cámara activa). El script dice cuál falta y guarda lo recibido en
  `camera_received.jpg`. Comprueba las 3 cámaras en la web de Station (:8889), el cable/puerto USB,
  y reinicia Station tras reconectar. La policy necesita las 3 vistas: no puede correr con 2.
- **`inference/normvla` no da frames:** el follower no tiene torque, o el leader también lo tiene
  (con `st3215-bus: "auto"`), o un servo del follower tiene el registro de estado distinto de 0:
  Station salta el frame entero (`Skip: motor_error` en su salida). Si pasa a mitad de un episodio,
  el script imprime el estado de cada servo antes de salir y marca el que tiene error. `normvla`
  es el nombre de la cola de Station, no del modelo: se usa con cualquier policy.
- **Station llena el disco:** graba siempre en `station_data/` (~2 GB/hora con 3 cámaras).
- **`station.yaml`:** `formats: []` funciona; una lista de strings da error de parseo.
- **Cámaras en Linux:** además del grupo `dialout`, hace falta una regla udev que dé acceso a
  `/dev/bus/usb` a las cámaras UVC (grupo `plugdev`).
- **`git sparse-checkout add` en norma-core** borra los protobufs generados si `target/gen_python`
  no está en el cone (`setup_env.sh` ya lo incluye).
- **`dataset-mp4` de NormaCore** falla con estos Parquet; para verlos, `inspect_normacore.py`.
- **Inferencia en CPU lenta:** limita los hilos con `POLICY_THREADS=4`.
