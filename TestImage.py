import os
import cv2
import numpy as np
from tensorflow import keras

def prediction(path):
    img_path = path
    
    model = keras.models.load_model('Logs/weeb_finder.keras')
    img = cv2.imread(img_path, 3)

    img = cv2.resize(img, (80, 80))
    img = np.reshape(img, [1, 80, 80, 3])

    pred = model.predict(img)

    return pred[0][1]
        
guess = prediction('test_weeb.png')
certainty = round(guess * 100, 2)
print(f"The AI is {certainty}% certain that the image contains a weeb")