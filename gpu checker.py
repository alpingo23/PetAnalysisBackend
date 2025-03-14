import tensorflow as tf
import os

# TensorFlow Sürümü
print("TensorFlow Sürümü:", tf.__version__)

# GPU Durumu
print("GPU'lar:", tf.config.list_physical_devices('GPU'))

# TensorFlow’un kullandığı CUDA ve cuDNN Sürümleri
try:
    sys_details = tf.sysconfig.get_build_info()
    print("TensorFlow CUDA Sürümü:", sys_details.get("cuda_version", "Bilinmiyor"))
    print("cuDNN Sürümü:", sys_details.get("cudnn_version", "Bilinmiyor"))
except Exception as e:
    print("Yapı bilgisi alınamadı:", e)

# Sistemde yüklü CUDA Sürümü (nvcc ile)
print("\nSistem CUDA Sürümü:")
os.system("nvcc --version")

# nvidia-smi Çıktısı
print("\nnvidia-smi Çıktısı:")
os.system("nvidia-smi")