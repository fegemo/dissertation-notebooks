import gc
import os
import numpy
import numpy as np
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy import asarray
from scipy.linalg import sqrtm
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from skimage.transform import resize
from skimage.io import imread


# scale an array of images to a new size
def _scale_images(images, new_shape):
    images_list = list()
    for image in images:
        # resize with nearest neighbor interpolation
        new_image = resize(image, new_shape, 0)
        # store
        images_list.append(new_image)
    return asarray(images_list)


def _calculate_fid(model, images1, images2, batch_size=136):
    act1 = np.empty((len(images1), 2048))
    act2 = np.empty((len(images2), 2048))

    gc.collect()
    for batch_start in range(0, len(images1), batch_size):
        batch_end = batch_start + batch_size
        batch_end = min(batch_end, len(images1))

        act1[batch_start:batch_end] = model.predict(images1[batch_start:batch_end], verbose=0)
        act2[batch_start:batch_end] = model.predict(images2[batch_start:batch_end], verbose=0)

    gc.collect()

    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
    # print("mu1, sigma1", mu1, sigma1)
    mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
    # print("mu2, sigma2", mu2, sigma2)
    # calculate sum squared difference between means
    ssdiff = numpy.sum((mu1 - mu2) ** 2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


def _load_directory_of_images(path):
    list_files = os.listdir(path)
    image_list = [imread(os.sep.join([path, filename])) for filename in list_files]
    return asarray(image_list)


def _compare_datasets(dataset1_path, dataset2_path, model):
    # loads the images from directories
    images1 = dataset1_path
    images2 = dataset2_path

    if type(dataset1_path) == str:
        images1 = _load_directory_of_images(dataset1_path)
    if type(dataset2_path) == str:
        images2 = _load_directory_of_images(dataset2_path)

    # resize images
    images1 = _scale_images(images1[..., 0:3], (299, 299, 3))
    images2 = _scale_images(images2[..., 0:3], (299, 299, 3))

    # pre-process images according to inception v3 expectations
    images1 = preprocess_input(images1)
    images2 = preprocess_input(images2)

    fid = _calculate_fid(model, images1, images2)
    return fid


inception_model = InceptionV3(include_top=False, pooling="avg", input_shape=(299, 299, 3))


def compare(dataset1_or_path, dataset2_or_path):
    return _compare_datasets(dataset1_or_path, dataset2_or_path, inception_model)
