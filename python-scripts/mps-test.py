import torch

# Verificar que MPS esté disponible
if torch.backends.mps.is_available():
    # Crear un dispositivo MPS
    mps_device = torch.device("mps")
    
    # Crear un tensor directamente en el GPU
    x = torch.ones(5, device=mps_device)
    y = x * 2  # La operación se ejecuta en el GPU
    
    # Mover tu modelo al GPU
    model = YourModel().to(mps_device)

