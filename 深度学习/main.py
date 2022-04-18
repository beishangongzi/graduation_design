# create by andy at 2022/3/28
# reference: https://www.tensorflow.org/tutorials/images/segmentation?hl=zh-cn
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

import models
from datasets import ObtDataset
from tensorflow_examples.models.pix2pix import pix2pix


def normalize(input_image, input_mask):
    input_image = tf.cast(input_image, tf.float32) / 1024.0
    return input_image, input_mask


@tf.function
def load_image_train(datapoint):
    input_image = tf.image.resize(datapoint['image'], (224, 224))
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (224, 224))

    # if tf.random.uniform(()) > 0.5:
    #     input_image = tf.image.flip_left_right(input_image)
    #     input_mask = tf.image.flip_left_right(input_mask)

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask

def load_image_test(datapoint):
  input_image = tf.image.resize(datapoint['image'], (224, 224))
  input_mask = tf.image.resize(datapoint['segmentation_mask'], (224, 224))

  input_image, input_mask = normalize(input_image, input_mask)

  return input_image, input_mask



def main():
    tr_dataset, tr_info = tfds.load("obt_dataset", split="train[:80%]", with_info=True)
    val_dataset, val_info = tfds.load("obt_dataset", split='train[80%:]', with_info=True)
    print(tr_info)
    TRAIN_LENGTH = 10 * 0.8
    BATCH_SIZE = 2
    BUFFER_SIZE = 10
    STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE

    train = tr_dataset.map(load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test = val_dataset.map(load_image_test)

    train_dataset = train.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    test_dataset = test.batch(BATCH_SIZE)

    OUTPUT_CHANNELS = 5

    model = models.FCN_Vgg16_32s(input_shape=(224, 224, 32))

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    tf.keras.utils.plot_model(model, show_shapes=True)
    EPOCHS = 20
    VAL_SUBSPLITS = 5
    # VALIDATION_STEPS = info.splits['test'].num_examples // BATCH_SIZE // VAL_SUBSPLITS

    model_history = model.fit(train_dataset, epochs=EPOCHS,
                              steps_per_epoch=STEPS_PER_EPOCH,
                              # validation_steps=VALIDATION_STEPS,
                              validation_data=test_dataset,
                              )
if __name__ == '__main__':
    main()
