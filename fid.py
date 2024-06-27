import gc
import logging

import numpy as np
import tensorflow as tf
from scipy.linalg import sqrtm

from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input

inception_model = None


def get_inception_model():
    global inception_model
    if inception_model is None:
        logging.info("Start >> Loading InceptionV3 model...")
        inception_model = InceptionV3(include_top=False, pooling="avg", input_shape=(299, 299, 3))
        logging.info("End   >> Loading InceptionV3 model.")
    return inception_model


def _prepare_images(images):
    images = tf.image.resize(images[..., 0:3], (299, 299), method="nearest")
    # preprocessing might not be necessary, but it's better to be safe
    images = preprocess_input(images)
    return images


def calculate_metrics_for_dataset(images, batch_size=128):
    model = get_inception_model()
    activations = np.empty((len(images), 2048))

    gc.collect()
    for batch_start in range(0, len(images), batch_size):
        batch_end = batch_start + batch_size
        batch_end = min(batch_end, len(images))

        batch = images[batch_start:batch_end]
        batch = _prepare_images(batch)
        activations[batch_start:batch_end] = model.predict(batch, verbose=0)
    gc.collect()

    mu, sigma = activations.mean(axis=0), np.cov(activations, rowvar=False)
    return mu, sigma


def calculate_fid_from_metrics(metrics_images_1, metrics_images_2):
    mu1, sigma1 = metrics_images_1
    mu2, sigma2 = metrics_images_2

    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


def calculate_fid(images_1, images_2):
    metrics_1 = calculate_metrics_for_dataset(images_1)
    metrics_2 = calculate_metrics_for_dataset(images_2)
    return calculate_fid_from_metrics(metrics_1, metrics_2)
