# uni-cognit-systems

Repositorio para experimentos de **sistemas cognitivos** usando **PyTorch** con aceleración por GPU (MPS) en macOS con Apple Silicon.

## Contenido del repositorio

```
uni-cognit-systems/
├── .envrc              # Configuración de direnv (activa el venv automáticamente)
├── .gitignore          # Archivos y carpetas excluidos del control de versiones
├── .python-version     # Versión de Python requerida (3.12)
├── requirements.txt    # Dependencias del proyecto
├── mps-test.py         # Script de prueba de aceleración MPS
├── notebooks/
│   └── test.ipynb      # Notebook de prueba
└── python-scripts/
    └── test.py         # Verifica disponibilidad de MPS
```

## Requisitos previos

- **Python 3.12** (se recomienda [`pyenv`](https://github.com/pyenv/pyenv) para gestionar versiones).
- **macOS con Apple Silicon** (chips M1/M2/M3/M4) para usar la aceleración MPS de PyTorch.
- (Opcional) [`direnv`](https://direnv.net/) para activar el entorno virtual automáticamente.

## Instalación

1. **Clona el repositorio** y entra en el directorio:

   ```bash
   git clone <url-del-repositorio>
   cd uni-cognit-systems
   ```

2. **Crea el entorno virtual**:

   ```bash
   python3.12 -m venv .venv
   ```

   > Si usas `pyenv`, puedes crear la versión con `pyenv install 3.12` y luego
   > `pyenv local 3.12` para que el entorno use la versión indicada en `.python-version`.

3. **Activa el entorno virtual**:

   ```bash
   source .venv/bin/activate
   ```

   > Alternativa con `direnv`: ejecuta `direnv allow` una sola vez y el entorno se
   > activará automáticamente al entrar en el directorio (gracias a `.envrc`).

4. **Instala las dependencias**:

   ```bash
   pip install -r requirements.txt
   ```

   Dependencias principales:

   | Paquete     | Propósito                                   |
   |-------------|---------------------------------------------|
   | `torch`     | Framework de aprendizaje profundo (PyTorch) |
   | `torchvision` | Visión por computadora                    |
   | `torchaudio` | Procesamiento de audio                     |
   | `ipykernel`  | Kernel de Python para Jupyter              |

## Ejecutar los notebooks

1. Con el entorno virtual activo, registra el kernel de Jupyter (solo la primera vez):

   ```bash
   python -m ipykernel install --user --name .venv --display-name ".venv (3.12.13)"
   ```

2. Inicia Jupyter:

   ```bash
   jupyter notebook notebooks/
   ```

   O, si prefieres la interfaz moderna:

   ```bash
   jupyter lab notebooks/
   ```

3. En la interfaz, abre `test.ipynb` y selecciona el kernel `.venv (3.12.13)`.

## Verificar la aceleración MPS (GPU)

El proyecto incluye scripts para comprobar que PyTorch puede usar la GPU de Apple Silicon:

```bash
# Con el entorno virtual activo
python python-scripts/test.py
```

Esto imprimirá `True` si MPS está disponible.

También puedes ejecutar el script de ejemplo:

```bash
python mps-test.py
```

> **Nota:** `mps-test.py` es una plantilla de ejemplo que muestra cómo mover tensores y
> modelos al dispositivo MPS (`torch.device("mps")`). El nombre `YourModel` es un
> marcador de posición que debes reemplazar por tu propio modelo.

## Notas

- **MPS** (Metal Performance Shaders) es el backend de PyTorch para aprovechar la GPU
  en Macs con Apple Silicon. Si tu máquina no lo soporta, PyTorch usará la CPU
  automáticamente.
- El entorno virtual (`.venv/`) y los checkpoints de Jupyter están excluidos del control
  de versiones mediante `.gitignore`.
