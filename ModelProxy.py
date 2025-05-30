import abc
import gc
import logging

import tensorflow as tf

import post_process
from post_process import batch_extract_palette, PostProcessor

DOMAINS = ["back", "left", "front", "right"]


class MultiInputTFSMLayer(tf.keras.layers.Layer):
    def __init__(self, model_path, input_names, output_names, legacy_endpoint, **kwargs):
        super(MultiInputTFSMLayer, self).__init__(**kwargs)
        self.model_path = model_path
        self.input_names = input_names
        self.output_names = output_names
        self.endpoint = "serving_default" if legacy_endpoint else "serve"
        self.model = tf.saved_model.load(model_path)

    def _get_signature(self, training=False):
        endpoint = self.endpoint + ("_training" if training else "")
        return self.model.signatures[endpoint]

    def call(self, inputs, training=False):
        callable_fn = self._get_signature(training)
        if self.input_names is not None:
            # stargan and collagan have input names associated, but pix2pix has only one input tensor
            inputs_dict = dict(zip(self.input_names, inputs))
            outputs = callable_fn(**inputs_dict)
        else:
            outputs = callable_fn(inputs)

        if len(outputs) == 1:
            return outputs[list(outputs.keys())[0]]
        else:
            return list(outputs.values())

    def predict(self, inputs, batch_size=4, training=False):
        # inputs is always a list of inputs, even if there is only one input tensor
        use_input_names = self.input_names is not None
        if use_input_names:
            inputs_dict = dict(zip(self.input_names, inputs))
        callable_fn = self._get_signature(training)
        outputs = []
        model_has_multiple_outputs = False
        batch_length = len(inputs[0])
        for i in range(0, batch_length, batch_size):
            if use_input_names:
                batch_inputs = {name: inputs_dict[name][i:i + batch_size] for name in self.input_names}
                batch_outputs = callable_fn(**batch_inputs)
            else:
                batch_end = min(i + batch_size, batch_length)
                batch_inputs = [input_tensor[i:batch_end] for input_tensor in inputs]
                batch_outputs = callable_fn(*batch_inputs)
            if len(batch_outputs) == 1:
                outputs.append(batch_outputs[list(batch_outputs.keys())[0]])
            else:
                # ignore the outputs that are not requested (absent from self.output_names)
                if self.output_names is not None:
                    batch_outputs = {name: batch_outputs[name] for name in self.output_names if name in batch_outputs}
                model_has_multiple_outputs = len(batch_outputs) > 1
                if not model_has_multiple_outputs:
                    outputs.append(batch_outputs.get(self.output_names[0]))
                else:
                    outputs.append(list(batch_outputs.values()))

        if model_has_multiple_outputs:
            # outputs is a list (len(outputs) == batch_length) in which each element is a list of outputs for that batch
            outputs = [tf.concat(output, axis=0) for output in zip(*outputs)]
            return outputs

        else:
            return tf.concat(outputs, axis=0)

    @property
    def layers(self):
        callable_fn = self._get_signature(training=False)
        print("callable_fn", callable_fn)
        print("dir(callable_fn)", dir(callable_fn))
        return self.callable_fn.layers

    def get_config(self):
        config = super(MultiInputTFSMLayer, self).get_config()
        config.update({
            "model_path": self.model_path,
            "input_names": self.input_names
        })
        return config


class ModelProxy(abc.ABC):
    def __init__(self, name, path, post_process, is_legacy_tf_saved_model_format=True):
        self.name = name
        self.post_process = post_process
        self.legacy_saved_model = is_legacy_tf_saved_model_format
        self.model = self._load_model(path)
        if post_process:
            self.post_processor = PostProcessor()

    @abc.abstractmethod
    def _load_model(self, path):
        pass

    def _load_saved_model(self, path, input_names=None, output_names=None):
        return MultiInputTFSMLayer(path, input_names=input_names, output_names=output_names,
                                   legacy_endpoint=self.legacy_saved_model)

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
            self.model = self._load_saved_model(self.model_paths[f"{source_name}-to-{target_name}"])
            # self.model = PostProcessGenerator(self.model, "cielab") if self.post_process else self.model
            self.loaded_model = f"{source_name}-to-{target_name}"
            logging.info(f"End   >> Loading Pix2Pix model {source_name}-to-{target_name}")
        return self.model

    def generate(self, source_domain, target_domain, batch):
        model = self._select_model(source_domain, target_domain)
        source_image = tf.gather(batch, source_domain)
        genned_image = model(source_image)
        if self.post_process:
            source_palette = batch_extract_palette(source_image)
            genned_image = self.post_processor.quantize_to_palette(genned_image, source_palette)
        return genned_image

    def encode(self, source_domain, target_domain, batch):
        model = self._select_model(source_domain, target_domain)
        model.summary(expand_nested=True)
        surrogate_model = tf.keras.Model(inputs=model.inputs, outputs=model.get_layer("sequential_6").outputs)
        source_image = tf.gather(batch, source_domain)
        encoded = surrogate_model(source_image)
        del surrogate_model
        return encoded

    def decode(self, source_domain, target_domain, code_batch):
        model = self._select_model(source_domain, target_domain)
        surrogate_model = tf.keras.Model(inputs=model.get_layer("sequential_6").input, outputs=model.output)
        decoded = surrogate_model(code_batch)
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


class IndexedPix2PixModelProxy(Pix2PixModelProxy):
    def __init__(self, path, post_process=False):
        super().__init__(path, post_process)

    @staticmethod
    def _single_extract_palette(image):
        # incoming image shape: (s, s, c)
        # reshaping to: (s*s, c)
        channels = 4
        image = tf.reshape(image, [-1, channels])

        # colors are sorted as they appear in the image sweeping from top-left to bottom-right
        colors, _ = tf.raw_ops.UniqueV2(x=image, axis=[0])

        # now sort according to their grayness value (this is the sorting done in the aiide paper)
        gray_coefficients = tf.constant([0.2989, 0.5870, 0.1140, 0.])[..., tf.newaxis]
        grayness = tf.squeeze(tf.matmul(tf.cast(colors, "float32"), gray_coefficients))
        indices_sorted_by_grayness = tf.argsort(grayness, direction="ASCENDING", stable=True)
        colors = tf.gather(colors, indices_sorted_by_grayness)

        return colors

    @staticmethod
    def _extract_palette(images):
        palettes_ragged = tf.map_fn(fn=IndexedPix2PixModelProxy._single_extract_palette, elems=images,
                                    fn_output_signature=tf.RaggedTensorSpec(
                                        ragged_rank=0,
                                        dtype=tf.int32))
        palettes = tf.RaggedTensor.to_tensor(palettes_ragged, default_value=tf.constant([255, 0, 255, 255],
                                                                                        dtype=tf.int32))
        return palettes

    @staticmethod
    def _single_rgba_to_indexed(image, palette):
        shape = tf.shape(image)
        s, c = shape[0], shape[2]
        flattened_image = tf.reshape(image, [s * s, c])
        num_pixels, num_components = s * s, c

        indices = flattened_image == palette[:, None]
        row_sums = tf.reduce_sum(tf.cast(indices, "int32"), axis=2)
        results = tf.cast(tf.where(row_sums == num_components), "int32")

        color_indices, pixel_indices = results[:, 0], results[:, 1]
        pixel_indices = tf.expand_dims(pixel_indices, -1)

        indexed = tf.scatter_nd(pixel_indices, color_indices, [num_pixels])
        indexed = tf.reshape(indexed, [shape[0], shape[1], 1])
        return indexed

    @staticmethod
    def _batch_rgba_to_indexed(batch, palettes):
        batch_shape = tf.shape(batch)
        d, b = batch_shape[0], batch_shape[1]
        indexed_batch = []
        for i in range(d):
            indexed_images = []
            for j in range(b):
                indexed_images.append(IndexedPix2PixModelProxy._single_rgba_to_indexed(batch[i][j], palettes[j]))
            indexed_batch.append(tf.stack(indexed_images, axis=0))
        indexed_batch = tf.stack(indexed_batch, axis=0)
        return indexed_batch

    @staticmethod
    def _indexed_to_rgba(indexed_images, palettes):
        b, s, _, _ = tf.shape(indexed_images)
        c = tf.shape(palettes)[-1]
        image_rgba = tf.gather(palettes, indexed_images, batch_dims=1)
        image_rgba = tf.reshape(image_rgba, [b, s, s, c])
        image_rgba = tf.cast(image_rgba, tf.float32) / 127.5 - 1.0
        return image_rgba

    def generate(self, source_domain, target_domain, batch):
        # batch comes as [-1, 1] float32 tensors...
        # convert the batch to int32 [0, 255] tensors
        batch = tf.cast((batch + 1.) * 127.5, tf.int32)

        # gets the source_image, so we can extract its palette
        source_image = tf.gather(batch, source_domain)
        target_image = tf.gather(batch, target_domain)
        combin_image = tf.concat([source_image, target_image], axis=-1)

        # extracts the int32 palette from the combined source and target image
        # palette = IndexedPix2PixModelProxy._extract_palette(combin_image)
        palette = IndexedPix2PixModelProxy._extract_palette(combin_image)

        # converts the batch to indexed images
        indexed_batch = IndexedPix2PixModelProxy._batch_rgba_to_indexed(batch, palette)
        indexed_batch = tf.cast(indexed_batch, tf.float32)

        # asks the regular Pix2PixModelProxy to generate the image, which will be a 64x64 indexed image
        genned_image_probabilities = super().generate(source_domain, target_domain, indexed_batch)

        # convert the probabilities to an indexed image
        genned_image = tf.argmax(genned_image_probabilities, axis=-1, output_type=tf.int32)[..., tf.newaxis]

        # convert the indexed image to an RGBA image in the range of [-1, 1] in float32
        genned_image = IndexedPix2PixModelProxy._indexed_to_rgba(genned_image, palette)

        return genned_image


class StarGANModelProxy(ModelProxy):
    def __init__(self, path, post_process=False, is_legacy_tf_saved_model_format=True):
        super().__init__("StarGAN", path, post_process, is_legacy_tf_saved_model_format)

    def _load_model(self, path):
        if self.legacy_saved_model and tf.__version__ < "2.16":
            return tf.keras.models.load_model(path, compile=False)
        else:
            return super()._load_saved_model(path, ["source_image", "target_domain", "source_domain"])

    def generate(self, source_domain, target_domain, batch):
        batch_shape = tf.shape(batch)
        batch_size = batch_shape[1]

        source_image = tf.gather(batch, source_domain)
        source_domain = tf.cast(source_domain, tf.float32)
        source_domain = tf.reshape(tf.tile(source_domain[tf.newaxis, ...], [batch_size, ]), [batch_size, 1])
        target_domain = tf.cast(target_domain, tf.float32)
        target_domain = tf.reshape(tf.tile(target_domain[tf.newaxis, ...], [batch_size, ]), [batch_size, 1])
        genned_image = self.model([source_image, target_domain, source_domain])
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
        return surrogate_model([source_image, target_domain, source_domain])

    def decode(self, source_domain, target_domain, code_batch):
        surrogate_model = tf.keras.Model(inputs=self.model.get_layer("add_5").input, outputs=self.model.output)
        return surrogate_model(code_batch)


class CollaGANModelProxy(ModelProxy):
    def __init__(self, path, palette_input=False, post_process=False, is_legacy_tf_saved_model=True,
                 domain_dtype="float32"):
        self.palette_input = palette_input
        self.gen_supplier = NParamsSupplier(2 if not palette_input else 3)
        self.temperature = None if palette_input else 0.
        self.domain_dtype = domain_dtype
        super().__init__("CollaGAN", path, post_process, is_legacy_tf_saved_model)

    def _load_model(self, path):
        m = self._load_saved_model(path, ["source_images", "target_domain"] + (
            ["desired_palette"] if self.palette_input else []))
        return m

    def set_temperature(self, temperature):
        self.temperature = temperature
        for v in [v for v in self.model.model.non_trainable_variables if "temperature" in v.name]:
            v.assign(temperature)

    # Generates images from a single source domain
    def generate(self, source_domain, target_domain, batch, training=False):
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

        target_domain = tf.cast(target_domain, self.domain_dtype)
        target_domain = tf.tile(target_domain[tf.newaxis, ...], [batch_size,])
        target_domain = tf.reshape(target_domain, [batch_size, 1])
        genned_image = self.model(self.gen_supplier(batch, target_domain,
                                                    lambda: post_process.batch_extract_palette(batch)),
                                  training=training)

        if self.post_process:
            source_palette = batch_extract_palette(source_image)
            genned_image = self.post_processor.quantize_to_palette(genned_image, source_palette)

        return genned_image

    def generate_from_multiple(self, target_domain, batch_transpose, training=False, target_palette=None):
        target_domain = tf.cast(target_domain, self.domain_dtype)
        target_domain = tf.reshape(target_domain, [-1, 1])
        genned_image = self.model(self.gen_supplier(batch_transpose, target_domain,
                                                    lambda: target_palette if
                                                    target_palette is not None else
                                                    post_process.batch_extract_palette(batch_transpose)),
                                  training=training)

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
        return surrogate_model(self.gen_supplier(batch, target_domain,
                                                 lambda: post_process.batch_extract_palette_ragged(batch)))

    def decode(self, source_domain, target_domain, code_batch):
        surrogate_model = tf.keras.Model(inputs=self.model.get_layer("re_lu_33").input, outputs=self.model.output)
        return surrogate_model(code_batch)


class RemicModelProxy(ModelProxy):
    def encode(self, source_domain, target_domain, batch):
        raise NotImplementedError("RemicModelProxy has not implemented encoding yet")
        # uce = self.model["unified_content_encoder"]
        # secs = self.model["style_encoders"]
        #
        # content_code = uce.predict(batch, verbose=0)
        # style_code = secs[source_domain].predict(batch, verbose=0)
        # return content_code, style_code

    def decode(self, source_domain, target_domain, code_batch):
        raise NotImplementedError("RemicModelProxy has not implemented decoding yet")

    def __init__(self, path, post_process=False):
        super().__init__("ReMIC", path, post_process)

    def _load_model(self, path):
        domain_initial_letters = [d[0].upper() for d in DOMAINS]
        paths = {
            "decoders": [f"{path}/decoders/Decoder{letter}" for letter in domain_initial_letters],
            "style_encoders": [f"{path}/style_encoders/StyleEncoder{letter}" for letter in domain_initial_letters],
            "unified_content_encoder": f"{path}/unified_content_encoder",
        }
        # each decoder has input of [(b, 8), (b, 16, 16, 256)] and output of (b, 64, 64, 4)
        # each style encoder has input of (b, 64, 64, 4) and output of (b, 8)
        # the unified content encoder has input of (b, d, 64, 64, 4) and output of (b, 16, 16, 256)
        # ste_input_names = [["input_7"], ["input_8"], ["input_9"], ["input_10"]]
        dec_input_names = [["input_13", "input_11"], ["input_16", "input_14"], ["input_19", "input_17"],
                           ["input_22", "input_20"]]
        dec_output_names = [["conv2d_231"], ["conv2d_242"], ["conv2d_253"], ["conv2d_264"]]
        # return {
        #     "decoders": [self._load_saved_model(model_path, dec_input_names[d]) for d, model_path in
        #                  enumerate(paths["decoders"])],
        #     "style_encoders": [self._load_saved_model(model_path, ste_input_names[d]) for d, model_path in
        #                        enumerate(paths["style_encoders"])],
        #     "unified_content_encoder": self._load_saved_model(paths["unified_content_encoder"], ["input_6"])
        # }
        return {
            # "decoders": [self._load_saved_model(model_path) for model_path in paths["decoders"]],
            "decoders": [self._load_saved_model(model_path, dec_input_names[d], dec_output_names[d]) for d, model_path
                         in
                         enumerate(paths["decoders"])],
            "style_encoders": [self._load_saved_model(model_path) for model_path in paths["style_encoders"]],
            "unified_content_encoder": self._load_saved_model(paths["unified_content_encoder"])
        }

    # Generates images from a single source domain to a single target domain
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

        bs = 4
        uce = self.model["unified_content_encoder"]
        sec = self.model["style_encoders"]
        dec = self.model["decoders"]
        content_code = uce.predict(batch, batch_size=bs, verbose=0)
        # content_code.shape is (b, 16, 16, 256)
        style_code = sec[source_domain].predict(batch, batch_size=bs, verbose=0)
        # style_code.shape is (b, 8)
        genned_image = dec[target_domain].predict([style_code, content_code], batch_size=bs, verbose=0)
        # genned_image.shape is (b, 64, 64, 4)

        if self.post_process:
            source_palette = batch_extract_palette(source_image)
            genned_image = self.post_processor.quantize_to_palette(genned_image, source_palette)

        return genned_image

    def generate_from_multiple(self, target_domain, batch_transpose, style_code=None):
        """
        :param target_domain: a tensor with the target domain with shape [b]
        :param batch_transpose: input images with shape [b, d, h, w, c] and at least one image dropped out
        for each example in the batch
        :param style_code: a tensor with hardcoded random style codes to be used with shape [b, 8]
        :return:
        """
        batch_size = tf.shape(target_domain)[0].numpy()
        uce = self.model["unified_content_encoder"]
        secs = self.model["style_encoders"]
        decs = self.model["decoders"]

        # there can be a different target_domain for each element in the batch -- hence, we cannot use a single
        # style encoder and decoder (we need to traverse the batch)...
        content_code = uce.predict([batch_transpose])
        # content_code.shape is (b, 16, 16, 256)

        if style_code is None:
            style_code = [secs[target_domain[b]].predict([batch_transpose[b, target_domain[b]][tf.newaxis, ...]])
                          for b in range(batch_size)]
        # style_code.shape is (b, 1, 8)

        genned_image = [decs[target_domain[b]].predict([content_code[b][tf.newaxis, ...], style_code[b]])
                        for b in range(batch_size)]
        genned_image = tf.concat(genned_image, axis=0)
        # genned_image.shape is (b, 64, 64, 4)

        return genned_image


class NParamsSupplier:
    def __init__(self, supply_first_n_params):
        self.n = supply_first_n_params

    def __call__(self, *args, **kwargs):
        return [arg() if callable(arg) else arg for arg in args[:self.n]]
