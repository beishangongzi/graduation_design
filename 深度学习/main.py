# create by andy at 2022/3/28
# reference: https://www.tensorflow.org/tutorials/images/segmentation?hl=zh-cn 

import tensorflow as tf
import tensorflow_datasets as tfds
from datasets import ObtDataset


def normalize(input_image, input_mask):
    input_image = tf.cast(input_image, tf.float32) / 1024.0
    return input_image, input_mask


@tf.function
def load_image_train(datapoint):
    input_image = tf.image.resize(datapoint['image'], (128, 128))
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))

    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask





def main():
    tr_dataset, tr_info = tfds.load("obt_dataset", split="train[:80%]", with_info=True)
    val_dataset, val_info = tfds.load("obt_dataset", split='train[80%:]', with_info=True)

    print(tr_info)
    print(val_info)


if __name__ == '__main__':
    main()
