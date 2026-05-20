import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

print("Loading model...")
model = tf.keras.models.load_model('model/efficientnetb3_dr.keras', compile=False)
print("Model loaded")

print("Loading thresholds...")
import numpy as np
thresholds = np.load('model/thresholds.npy')
print("Thresholds loaded")

print("Importing fuzzy modules...")
from model.fuzzy_orient import OrientationFuzzy
from model.fuzzy_adj import AdjustmentFuzzy
print("All imports successful")

print("Test passed.")