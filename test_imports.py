import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("Step 1: Importing TensorFlow...")
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

print("Step 2: Loading model...")
model = tf.keras.models.load_model('model/efficientnetb3_dr.keras', compile=False)
print("Model loaded OK")

print("Step 3: Loading numpy...")
import numpy as np

print("Step 4: Loading thresholds...")
thresholds = np.load('model/thresholds.npy')
print("Thresholds:", thresholds)

print("Step 5: Importing cv2...")
import cv2

print("Step 6: Importing fuzzy modules...")
from model.fuzzy_orient import OrientationFuzzy
from model.fuzzy_adj import AdjustmentFuzzy
print("All imports successful")

print("Step 7: Creating Flask app...")
from flask import Flask
app = Flask(__name__)

print("Step 8: Running server (will block)...")
app.run(debug=True, host='0.0.0.0', port=5000)