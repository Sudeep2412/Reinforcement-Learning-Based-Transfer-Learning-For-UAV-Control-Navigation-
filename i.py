import tensorflow as tf

print("TensorFlow version:", tf.__version__)
gpus = tf.config.list_physical_devices('GPU')

if gpus:
    print("✅ GPU(s) available:")
    for gpu in gpus:
        print("  -", gpu)
    # Enable memory growth (optional but recommended)
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("❌ No GPU available. TensorFlow is running on CPU.")
