import tensorflow as tf
from fid import calculate_fid_from_metrics, get_inception_model, calculate_metrics_for_dataset


def preload():
    get_inception_model()


def partial_fid_metrics(images):
    return calculate_metrics_for_dataset(images)


# Calculates the FID between two image datasets, returning a Numpy float
# @param images_a a dataset of images as a tensor
# @param images_b a dataset of images as a tensor
# @return the FID between the two datasets as a Numpy float
def calculate_fid(metrics_a, metrics_b):
    return calculate_fid_from_metrics(metrics_a, metrics_b)


# Calculates the L1 distance between two image datasets, returning a Numpy float
# @param images_a a dataset of images as a tensor
# @param images_b a dataset of images as a tensor
# @return the L1 distance between the two datasets as a Numpy float
@tf.function
def calculate_l1(images_a, images_b):
    return tf.reduce_mean(tf.abs(images_b - images_a))
