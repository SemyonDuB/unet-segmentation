import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


def normalize(image):
    return tf.cast(image, tf.float32) / 255.0


def load_img(img):
    image = tf.image.resize(img, (256, 256), method="nearest")
    image = normalize(image)
    return tf.expand_dims(image, 0)


def parse_image(img_path: str):
    image = tf.io.read_file(img_path)
    return tf.io.decode_jpeg(image)


def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]


def display(display_list):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()


def iou(y_true, y_pred):
    def f(y_true, y_pred):
        intersection = (y_true * y_pred).sum()
        union = y_true.sum() + y_pred.sum() - intersection
        x = (intersection + 1e-15) / (union + 1e-15)
        x = x.astype(np.float32)
        return x

    return tf.numpy_function(f, [y_true, y_pred], tf.float32)


images = ["./a.jpg", "./b.jpg", "./c.jpg", "./d.jpg", "./e.jpg"]
data = [load_img(parse_image(image)) for image in images]

model = tf.keras.models.load_model('my-model.keras', custom_objects={'iou': iou})

for image in data:
    predicted = model.predict(image)
    display([image[0, :, :, :], create_mask(predicted)])
