import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from flask import Flask
import numpy as np
from model.fuzzy_orient import OrientationFuzzy
from model.fuzzy_adj import AdjustmentFuzzy

app = Flask(__name__)

print("Loading model...")
model = tf.keras.models.load_model('model/efficientnetb3_dr.keras', compile=False)
print("✓ Model loaded")

print("Loading thresholds...")
thresholds = np.load('model/thresholds.npy')
print("✓ Thresholds loaded:", thresholds)

print("Loading fuzzy modules...")
orient = OrientationFuzzy()
adj = AdjustmentFuzzy()
print("✓ All modules loaded")

@app.route('/')
def home():
    return "Flask is running – your backend is alive!"

if __name__ == '__main__':
    print("\n Starting Flask server on http://0.0.0.0:5000 ...\n")
    app.run(debug=True, host='0.0.0.0', port=5000)