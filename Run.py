import cv2
import tensorflow as tf

CATEGORIES = ["X", "1", "2", "3", "4", "5", "6"]

def prepare(filepath): #turn image into 2D array that goes into network
    IMG_SIZE = 100  # 50 in txt-based
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)


model = tf.keras.models.load_model("Neural_Network_Proj_2.model")

prediction = model.predict([prepare('C:/Users/Aleksandar/Desktop/Neural_Network_Proj_2/TEST/1.jpg')])
print(CATEGORIES[int(prediction[0][0])])