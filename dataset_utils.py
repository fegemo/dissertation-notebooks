import math
import tensorflow as tf

DATASET_SIZES = [912, 216, 294, 408, 12372]
DATASET_NAMES = ["tiny-hero", "rpg-maker-2000", "rpg-maker-xp", "rpg-maker-vxace", "miscellaneous"]

starting_test_sample_numbers = stsn = {"tiny": 0, "rm2k": 136, "rmxp": 168, "rmvx": 212, "misc": 273}


# Some images have transparent pixels with colors other than black
# This function turns all transparent pixels to black
def blacken_transparent_pixels(image):
    mask = tf.math.equal(image[:, :, 3], 0)
    repeated_mask = tf.repeat(mask, 4)
    condition = tf.reshape(repeated_mask, image.shape)

    image = tf.where(
        condition,
        image * 0.,
        image * 1.)
    return image


# Loads an image from the specified path in the format required by the models
# @param path the path to the image (string)
def load_image(path, image_size=64, channels=4):
    image = None
    try:
        image = tf.io.read_file(path)
        image = tf.image.decode_png(image, channels=channels)
        image = tf.reshape(image, [image_size, image_size, channels])
        image = tf.cast(image, tf.float32)
        image = blacken_transparent_pixels(image)
        image = (image / 127.5) - 1
    except Exception as e:
        print(f"Error opening image in {path}.", e)

    return image


class DatasetLoader:
    def __init__(self, dataset, train_or_test="test", limit=None):
        self.dataset_name = dataset
        self.train_or_test = train_or_test
        self.dataset_sizes = self._calculate_dataset_sizes()
        self.dataset = (tf.data.Dataset.range(self.dataset_size if limit is None else limit)
                        .map(lambda image_number: self.load_paired_images(image_number),
                             num_parallel_calls=tf.data.AUTOTUNE)
                        .prefetch(tf.data.AUTOTUNE))

    def _calculate_dataset_sizes(self):
        dataset_mask = [1 if self.dataset_name == name else 0 for name in DATASET_NAMES]
        if self.dataset_name == "all":
            dataset_mask = [1, 1, 1, 1, 1]

        masked_dataset_sizes = [n * m for n, m in zip(DATASET_SIZES, dataset_mask)]
        train_sizes = [math.ceil(n * 0.85) for n in masked_dataset_sizes]
        test_sizes = [masked_dataset_sizes[i] - train_sizes[i] for i, n in enumerate(masked_dataset_sizes)]
        return train_sizes if self.train_or_test == "train" else test_sizes

    # Loads a sample of paired images (back, left, front, right) from the dataset
    # @param index the index of the image to load (int)
    @tf.function
    def load_paired_images(self, image_number):
        dataset_index = tf.constant(0, dtype=tf.int32)
        image_number = tf.cast(image_number, tf.int32)

        condition = lambda which_image, which_dataset: which_image >= tf.gather(self.dataset_sizes, which_dataset)
        body = lambda which_image, which_dataset: [which_image - tf.gather(self.dataset_sizes, which_dataset),
                                                   which_dataset + 1]
        image_number, dataset_index = tf.while_loop(condition, body, [image_number, dataset_index])

        dataset_folder = tf.gather(DATASET_NAMES, dataset_index)
        image_number = tf.strings.as_string(image_number)

        # finds the path and load the paired images
        paths = [tf.strings.join(["datasets/", dataset_folder, "/", self.train_or_test, "/", domain, "/", image_number, ".png"]) for
                 domain in ["0-back", "1-left", "2-front", "3-right"]]
        paired_images = [load_image(path) for path in paths]

        return paired_images

    @property
    def dataset_size(self):
        return sum(self.dataset_sizes)


