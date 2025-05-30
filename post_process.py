import tensorflow as tf
from scipy.spatial import KDTree
from tensorflow import RaggedTensorSpec

INVALID_COLOR = tf.constant([32768, 32768, 32768, 32768])


# function rgb_to_xyz extracted from tensorflow_io.experimental.color, from the url:
# https://github.com/tensorflow/io/blob/v0.24.0/tensorflow_io/python/experimental/color_ops.py#L333-L360
def rgb_to_xyz(input_tensor, name=None):
    """
    Convert a RGB image to CIE XYZ.

    Args:
      input_tensor: A 3-D (`[H, W, 3]`) or 4-D (`[N, H, W, 3]`) Tensor.
      name: A name for the operation (optional).

    Returns:
      A 3-D (`[H, W, 3]`) or 4-D (`[N, H, W, 3]`) Tensor.
    """
    input_tensor = tf.convert_to_tensor(input_tensor)
    assert input_tensor.dtype in (tf.float16, tf.float32, tf.float64)

    kernel = tf.constant(
        [
            [0.412453, 0.357580, 0.180423],
            [0.212671, 0.715160, 0.072169],
            [0.019334, 0.119193, 0.950227],
        ],
        input_tensor.dtype,
    )
    value = tf.where(
        tf.math.greater(input_tensor, 0.04045),
        tf.math.pow((input_tensor + 0.055) / 1.055, 2.4),
        input_tensor / 12.92,
    )
    return tf.tensordot(value, tf.transpose(kernel), axes=((-1,), (0,)))

# function rgb_to_lab extracted from tensorflow_io.experimental.color, from the url:
# https://github.com/tensorflow/io/blob/v0.24.0/tensorflow_io/python/experimental/color_ops.py#L398-L459
def rgb_to_lab(input_tensor, illuminant="D65", observer="2", name=None):
    """
    Convert a RGB image to CIE LAB.

    Args:
      input_tensor: A 3-D (`[H, W, 3]`) or 4-D (`[N, H, W, 3]`) Tensor.
      illuminant : {"A", "D50", "D55", "D65", "D75", "E"}, optional
        The name of the illuminant (the function is NOT case sensitive).
      observer : {"2", "10"}, optional
        The aperture angle of the observer.
      name: A name for the operation (optional).

    Returns:
      A 3-D (`[H, W, 3]`) or 4-D (`[N, H, W, 3]`) Tensor.
    """
    input_tensor = tf.convert_to_tensor(input_tensor)
    assert input_tensor.dtype in (tf.float16, tf.float32, tf.float64)

    illuminants = {
        "A": {
            "2": (1.098466069456375, 1, 0.3558228003436005),
            "10": (1.111420406956693, 1, 0.3519978321919493),
        },
        "D50": {
            "2": (0.9642119944211994, 1, 0.8251882845188288),
            "10": (0.9672062750333777, 1, 0.8142801513128616),
        },
        "D55": {
            "2": (0.956797052643698, 1, 0.9214805860173273),
            "10": (0.9579665682254781, 1, 0.9092525159847462),
        },
        "D65": {
            "2": (0.95047, 1.0, 1.08883),
            "10": (0.94809667673716, 1, 1.0730513595166162),
        },
        "D75": {
            "2": (0.9497220898840717, 1, 1.226393520724154),
            "10": (0.9441713925645873, 1, 1.2064272211720228),
        },
        "E": {"2": (1.0, 1.0, 1.0), "10": (1.0, 1.0, 1.0)},
    }
    coords = tf.constant(illuminants[illuminant.upper()][observer], input_tensor.dtype)

    xyz = rgb_to_xyz(input_tensor)

    xyz = xyz / coords

    xyz = tf.where(
        tf.math.greater(xyz, 0.008856),
        tf.math.pow(xyz, 1.0 / 3.0),
        xyz * 7.787 + 16.0 / 116.0,
    )

    xyz = tf.unstack(xyz, axis=-1)
    x, y, z = xyz[0], xyz[1], xyz[2]

    # Vector scaling
    l = (y * 116.0) - 16.0
    a = (x - y) * 500.0
    b = (y - z) * 200.0

    return tf.stack([l, a, b], axis=-1)


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


def extract_palette_ragged(image):
    """
    Extracts the unique colors from an image (3D tensor) -- returns a ragged tensor with shape [1, (colors), c]
    :params: image: a 3D tensor with shape (height, width, channels). Values should be inside [0, 255].
    :returns: a ragged tensor with shape [1, (num_colors), channels] with the palette colors as uint8 inside [0, 255]
    """
    channels = tf.shape(image)[-1]

    # incoming image shape: (s, s, channels)
    # reshaping to: (s*s, channels)
    image = tf.cast(image, tf.uint8)
    image = tf.reshape(image, [-1, channels])

    # colors are sorted as they appear in the image sweeping from top-left to bottom-right
    colors, _ = tf.raw_ops.UniqueV2(x=image, axis=[0])
    colors = tf.reshape(colors, [-1, channels])
    number_of_colors = tf.shape(colors)[0]
    # turns colors (a regular [num_colors, channels] tensor into a ragged tensor [1, (num_colors), channels])
    # this is necessary for an outer map_fn to call this function and have a ragged tensor as output
    colors = tf.RaggedTensor.from_row_lengths(values=colors, row_lengths=[number_of_colors])
    return colors


@tf.function
def batch_extract_palette_ragged(images):
    """
    Extracts the palette of each image in the batch, returning a ragged tensor of shape [b, (colors), c]
    :param images: batch of images: [b, s, s, c] with values inside [-1, 1]
    :return: a ragged tensor with shape [b, (colors), c] with the palette colors as float32 inside [-1, 1]
    """
    images = (images + 1) * 127.5
    images = tf.cast(images, tf.uint8)
    channels = images.shape[-1]

    palettes_ragged = tf.map_fn(fn=extract_palette_ragged, elems=images,
                                fn_output_signature=RaggedTensorSpec(
                                    shape=(1, None, channels),
                                    ragged_rank=1,
                                    dtype=tf.uint8))

    palettes_ragged = palettes_ragged.merge_dims(1, 2)
    palettes_ragged = tf.cast(palettes_ragged, tf.float32)
    palettes_ragged = (palettes_ragged / 127.5) - 1

    return palettes_ragged


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
        batch_image_lab = rgb_to_lab(batch_image_rgb)
        batch_image = tf.concat([batch_image_lab, batch_image_alpha], -1)
        batch_palette_lab = rgb_to_lab(batch_palette_rgb)
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
