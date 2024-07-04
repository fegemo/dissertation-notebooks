import abc
import gc
import logging

import tensorflow as tf

from post_process import batch_extract_palette, PostProcessor

DOMAINS = ["back", "left", "front", "right"]


class ModelProxy(abc.ABC):
    def __init__(self, name, path, post_process):
        self.name = name
        self.post_process = post_process
        self.model = self._load_model(path)
        if post_process:
            self.post_processor = PostProcessor()

    @abc.abstractmethod
    def _load_model(self, path):
        pass

    # Generates an image by translating it from the source_domain into the target_domain
    # @param source_domain the index of the source domain
    # @param target_domain the index of the target domain
    # @param batch the batch of images to translate in the format of a (back, left, front, right) batch. That is, it
    # has a shape of [d, b, h, w, c] where d is the number of domains, b is the batch size, h is the height, w is the
    def generate(self, source_domain, target_domain, batch):
        return

    @abc.abstractmethod
    def encode(self, source_domain, target_domain, batch):
        pass

    @abc.abstractmethod
    def decode(self, source_domain, target_domain, code_batch):
        pass


class Pix2PixModelProxy(ModelProxy):
    def __init__(self, path, post_process=False):
        super().__init__("Pix2Pix", path, post_process)
        self.model_paths = Pix2PixModelProxy.get_model_paths(path)

    def _load_model(self, path):
        self.loaded_model = ""
        return None

    def _select_model(self, source_domain, target_domain):
        source_name = DOMAINS[source_domain]
        target_name = DOMAINS[target_domain]
        if self.loaded_model != f"{source_name}-to-{target_name}":
            del self.model
            gc.collect()
            logging.info(f"Start >> Loading Pix2Pix model {source_name}-to-{target_name}")
            self.model = tf.keras.models.load_model(self.model_paths[f"{source_name}-to-{target_name}"], compile=False)
            # self.model = PostProcessGenerator(self.model, "cielab") if self.post_process else self.model
            self.loaded_model = f"{source_name}-to-{target_name}"
            logging.info(f"End   >> Loading Pix2Pix model {source_name}-to-{target_name}")
        return self.model

    def generate(self, source_domain, target_domain, batch):
        model = self._select_model(source_domain, target_domain)
        source_image = tf.gather(batch, source_domain)
        genned_image = model(source_image, training=True)
        if self.post_process:
            source_palette = batch_extract_palette(source_image)
            genned_image = self.post_processor.quantize_to_palette(genned_image, source_palette)
        return genned_image

    def encode(self, source_domain, target_domain, batch):
        model = self._select_model(source_domain, target_domain)
        model.summary(expand_nested=True)
        surrogate_model = tf.keras.Model(inputs=model.inputs, outputs=model.get_layer("sequential_6").outputs)
        source_image = tf.gather(batch, source_domain)
        encoded = surrogate_model(source_image, training=True)
        del surrogate_model
        return encoded

    def decode(self, source_domain, target_domain, code_batch):
        model = self._select_model(source_domain, target_domain)
        surrogate_model = tf.keras.Model(inputs=model.get_layer("sequential_6").input, outputs=model.output)
        decoded = surrogate_model(code_batch, training=True)
        del surrogate_model
        return decoded

    def __del__(self):
        del self.model
        gc.collect()

    @staticmethod
    def get_model_paths(base_path):
        return {
            "back-to-back": f"{base_path}/back-to-back",
            "back-to-left": f"{base_path}/back-to-left",
            "back-to-front": f"{base_path}/back-to-front",
            "back-to-right": f"{base_path}/back-to-right",
            "left-to-back": f"{base_path}/left-to-back",
            "left-to-left": f"{base_path}/left-to-left",
            "left-to-front": f"{base_path}/left-to-front",
            "left-to-right": f"{base_path}/left-to-right",
            "front-to-back": f"{base_path}/front-to-back",
            "front-to-left": f"{base_path}/front-to-left",
            "front-to-front": f"{base_path}/front-to-front",
            "front-to-right": f"{base_path}/front-to-right",
            "right-to-back": f"{base_path}/right-to-back",
            "right-to-left": f"{base_path}/right-to-left",
            "right-to-front": f"{base_path}/right-to-front",
            "right-to-right": f"{base_path}/right-to-right",
        }


class StarGANModelProxy(ModelProxy):
    def __init__(self, path, post_process=False):
        super().__init__("StarGAN", path, post_process)

    def _load_model(self, path):
        return tf.keras.models.load_model(path, compile=False)

    def generate(self, source_domain, target_domain, batch):
        batch_shape = tf.shape(batch)
        batch_size = batch_shape[1]

        source_image = tf.gather(batch, source_domain)
        source_domain = tf.constant(source_domain, tf.int32)
        source_domain = tf.tile(source_domain[tf.newaxis, ...], [batch_size, ])
        target_domain = tf.constant(target_domain, tf.int32)
        target_domain = tf.tile(target_domain[tf.newaxis, ...], [batch_size, ])
        # genned_image = self.model([source_image, target_domain], training=True)
        genned_image = self.model([source_image, target_domain, source_domain], training=True)
        if self.post_process:
            source_palette = batch_extract_palette(source_image)
            genned_image = self.post_processor.quantize_to_palette(genned_image, source_palette)
        return genned_image

    def encode(self, source_domain, target_domain, batch):
        batch_shape = tf.shape(batch)
        batch_size = batch_shape[1]

        source_image = tf.gather(batch, source_domain)
        source_domain = tf.constant(source_domain, tf.int32)
        source_domain = tf.tile(source_domain[tf.newaxis, ...], [batch_size, ])
        target_domain = tf.constant(target_domain, tf.int32)
        target_domain = tf.tile(target_domain[tf.newaxis, ...], [batch_size, ])
        surrogate_model = tf.keras.Model(inputs=self.model.input, outputs=self.model.get_layer("add_5").output)
        return surrogate_model([source_image, target_domain, source_domain], training=True)

    def decode(self, source_domain, target_domain, code_batch):
        surrogate_model = tf.keras.Model(inputs=self.model.get_layer("add_5").input, outputs=self.model.output)
        return surrogate_model(code_batch, training=True)


class CollaGANModelProxy(ModelProxy):
    def __init__(self, path, post_process=False):
        super().__init__("CollaGAN", path, post_process)

    def _load_model(self, path):
        return tf.keras.models.load_model(path, compile=False)

    # Generates images from a single source domain
    def generate(self, source_domain, target_domain, batch):
        batch_shape = tf.shape(batch)
        number_of_domains, batch_size = len(DOMAINS), batch_shape[1]

        # so we can do the post process, let's keep the source image
        source_image = batch[source_domain]

        keep_image_mask = tf.constant([1 if i == source_domain else 0 for i in range(number_of_domains)])
        # keep_image_mask.shape is [d]
        keep_image_mask = tf.tile(keep_image_mask[..., tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis],
                                  [1, batch_size, 1, 1, 1])
        # keep_image_mask.shape is [d, b, 1, 1, 1]
        batch *= tf.cast(keep_image_mask, tf.float32)

        # rearranges the batch so that the batch dimension becomes the first one
        batch = tf.transpose(batch, [1, 0, 2, 3, 4])
        # batch.shape is [b, d, h, w, c]

        target_domain = tf.constant(target_domain, tf.int32)
        target_domain = tf.tile(target_domain[tf.newaxis, ...], [batch_size, ])
        genned_image = self.model([batch, target_domain], training=True)

        if self.post_process:
            source_palette = batch_extract_palette(source_image)
            genned_image = self.post_processor.quantize_to_palette(genned_image, source_palette)

        return genned_image

    def generate_from_multiple(self, target_domain, batch_transpose):
        genned_image = self.model([batch_transpose, target_domain], training=True)
        if self.post_process:
            # batch_transpose has shape (b, d, h, w, c)
            batch_shape = tf.shape(batch_transpose)
            # we want to make it (b, d*h, w, c)
            source_images = tf.reshape(batch_transpose, [batch_shape[0], -1, batch_shape[3], batch_shape[4]])
            source_palette = batch_extract_palette(source_images)
            genned_image = self.post_processor.quantize_to_palette(genned_image, source_palette)

        return genned_image

    def encode(self, source_domain, target_domain, batch):
        batch_shape = tf.shape(batch)
        number_of_domains, batch_size = len(DOMAINS), batch_shape[1]

        # so we can do the post process, let's keep the source image
        source_image = batch[source_domain]

        keep_image_mask = tf.constant([1 if i == source_domain else 0 for i in range(number_of_domains)])
        # keep_image_mask.shape is [d]
        keep_image_mask = tf.tile(keep_image_mask[..., tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis],
                                  [1, batch_size, 1, 1, 1])
        # keep_image_mask.shape is [d, b, 1, 1, 1]
        batch *= tf.cast(keep_image_mask, tf.float32)

        # rearranges the batch so that the batch dimension becomes the first one
        batch = tf.transpose(batch, [1, 0, 2, 3, 4])
        # batch.shape is [b, d, h, w, c]

        target_domain = tf.constant(target_domain, tf.int32)
        target_domain = tf.tile(target_domain[tf.newaxis, ...], [batch_size, ])
        surrogate_model = tf.keras.Model(inputs=self.model.input, outputs=self.model.get_layer("re_lu_33").output)
        return surrogate_model([batch, target_domain], training=True)

    def decode(self, source_domain, target_domain, code_batch):
        surrogate_model = tf.keras.Model(inputs=self.model.get_layer("re_lu_33").input, outputs=self.model.output)
        return surrogate_model(code_batch, training=True)
