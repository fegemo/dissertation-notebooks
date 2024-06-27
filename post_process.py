import tensorflow as tf
import tensorflow_io as tfio
from scipy.spatial import KDTree

INVALID_COLOR = tf.constant([32768, 32768, 32768, 32768])


def batch_extract_palette(images):
    def single_extract_palette(image):
        channels = tf.shape(image)[-1]

        # incoming image shape: (s, s, channels)
        # reshaping to: (s*s, channels)
        image = tf.cast(image, "int32")
        image = tf.reshape(image, [-1, channels])

        # colors are sorted as they appear in the image sweeping from top-left to bottom-right
        colors, _ = tf.raw_ops.UniqueV2(x=image, axis=[0])
        return colors

    """
    Extracts the palette of each image in the batch, returning a ragged tensor of shape [b, (colors), c]
    :param images:
    :return:
    """
    images = (images + 1) * 127.5
    images = tf.cast(images, tf.int32)

    palettes_ragged = tf.map_fn(fn=single_extract_palette, elems=images,
                                fn_output_signature=tf.RaggedTensorSpec(
                                    ragged_rank=0,
                                    dtype=tf.int32))
    palettes = tf.RaggedTensor.to_tensor(palettes_ragged, default_value=INVALID_COLOR)
    palettes = tf.cast(palettes, tf.float32)
    palettes = (palettes / 127.5) - 1

    return palettes


class PostProcessor:
    @staticmethod
    def quantize_to_palette(batch_image, batch_palette):
        # batch_image and batch_palette come in [-1, 1]
        batch_palette_original = batch_palette
        # but they must be in [0, 1] for conversion to lab/yuv
        batch_image = batch_image * 0.5 + 0.5
        batch_palette = batch_palette * 0.5 + 0.5

        batch_image_rgb = batch_image[..., :3]
        batch_image_alpha = batch_image[..., 3:]
        batch_palette_rgb = batch_palette[..., :3]
        batch_palette_alpha = batch_palette[..., 3:]
        batch_image_lab = tfio.experimental.color.rgb_to_lab(batch_image_rgb)
        batch_image = tf.concat([batch_image_lab, batch_image_alpha], -1)
        batch_palette_lab = tfio.experimental.color.rgb_to_lab(batch_palette_rgb)
        batch_palette = tf.concat([batch_palette_lab, batch_palette_alpha], -1)

        batch_image = batch_image.numpy()
        batch_palette = batch_palette.numpy()
        batch_palette_original = batch_palette_original.numpy()

        results = []
        for image, palette, palette_original in zip(batch_image, batch_palette, batch_palette_original):
            # creates a tree of similar colors
            palette_tree = KDTree(palette)
            # finds the closest color index for each pixel
            _, indices = palette_tree.query(image)
            # creates the image quantized to the palette
            result = palette_original[indices]
            # adds the just palette-quantized image to the results batch
            results.append(result)

        results = tf.stack(results)
        return results
