import tensorflow as tf

# Verificar si TensorFlow detecta una GPU disponible
gpus = tf.config.list_physical_devices('GPU')
cpus = tf.config.list_physical_devices('CPU')

print('TensorFlow version:', tf.__version__)
print('GPU devices:', gpus)
print('CPU devices:', cpus)

if gpus:
    print('=> GPU disponible para TensorFlow/Keras.')
    for gpu in gpus:
        print('   -', gpu)
else:
    print('=> No hay GPU disponible: Keras/TensorFlow entrenará en CPU.')
