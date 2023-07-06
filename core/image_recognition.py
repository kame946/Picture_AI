import tensorflow as tf
import numpy as np

class ImageRecognitionModel:
    def __init__(self):
        self.model = tf.keras.applications.MobileNetV2()
        self.class_names = tf.keras.applications.mobilenet_v2.decode_predictions

    def predict_image(self, image):
        image = tf.keras.preprocessing.image.img_to_array(image)
        image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
        image = np.expand_dims(image, axis=0)
        predictions = self.model.predict(image)
        top_predictions = self.class_names(predictions, top=3)[0]
        return [(label, str(round(probability * 100, 2)) + '%') for (_, label, probability) in top_predictions]
