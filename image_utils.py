import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

DOMAINS = ["back", "left", "front", "right"]


def show_single_image(image, title=""):
    if title != "":
        plt.title(title)
    plt.imshow(image * 0.5 + 0.5)
    plt.axis("off")
    plt.show()


def show_batch(batch):
    batch_shape = tf.shape(batch)
    batch_size = batch_shape[1]

    batch = batch * 0.5 + 0.5
    for i in range(batch.shape[0]):
        plt.subplot(1, batch_size, i + 1)
        plt.imshow(batch[i])
        plt.axis("off")
        plt.show()


def show_comparison(source_images, target_images, genned_images, target_indices, column_titles, row_titles=None,
                    title="", suppress_target=False):
    """
    Generic function to display images side by side, with columns showing the source, target, and the generated images.
    The rows depict the different characters in the batch. The images received are in the range of [-1, 1].
    :param source_images: a tensor of shape (batch_size, 64, 64, 4) or (batch_size, domains, 64, 64, 4)
    :param target_images: a tensor of shape (batch_size, 64, 64, 4)
    :param genned_images: a list with n tensors of shape (batch_size, 64, 64, 4) or (batch_size, domains, 64, 64, 4)
    :param target_indices: a tensor of shape (batch_size) representing the index of the target domain
    :param column_titles: a list of strings with the titles of each column other than Source and Target
    :param row_titles: a list of strings to be displayed on the left side of the plot for each row
    :param title: the title of the plot (optional, default="")
    :param suppress_target: boolean indicating whether the target column should be suppressed
    :return: the figure object created by matplotlib
    """
    batch_shape = tf.shape(source_images)
    batch_size = batch_shape[0].numpy()
    cols = len(column_titles) + 1 + (0 if suppress_target else 1)
    rows = batch_size

    source_images = source_images * 0.5 + 0.5
    target_images = target_images * 0.5 + 0.5
    genned_images = [genned_images[i] * 0.5 + 0.5 for i in range(len(genned_images))]

    half_domains = len(DOMAINS) // 2
    if not suppress_target:
        column_titles = ["Source", "Target", *column_titles[:len(genned_images)]]
        column_contents = [source_images, target_images, *genned_images]
    else:
        column_titles = ["Source", *column_titles[:len(genned_images)]]
        column_contents = [source_images, *genned_images]

    fig = plt.figure(figsize=(8 * cols, 8 * rows))
    plt.suptitle(title)
    is_subplot_cell = lambda content: len(tf.shape(content)) == 4 and tf.shape(content)[0] == len(DOMAINS)

    sub_figs = fig.subfigures(rows, cols)
    for i in range(rows):
        for j in range(cols):
            sub_figs[i, j].suptitle(column_titles[j] if i == 0 else "", fontsize=40)
            sub_figs[i, j].patch.set_alpha(0.0)

            column_content = column_contents[j]
            cell_content = column_content[i]

            if not is_subplot_cell(cell_content):
                # not a subplot, only a single image to be displayed
                ax = sub_figs[i, j].subplots(1, 1)
                ax.imshow(cell_content, interpolation="nearest")
                ax.axis("off")
            else:
                # multiple images (3 or 4) to display, so we need subplots
                axes = sub_figs[i, j].subplots(half_domains, half_domains)
                for i_d in range(half_domains):
                    for j_d in range(half_domains):
                        ax = axes[i_d, j_d]
                        image = cell_content[i_d * half_domains + j_d]
                        if tf.math.count_nonzero(image) == 0:
                            # no image (either the target or a dropped image in the sources column)
                            target_index = target_indices[i]
                            text_to_show = "Target" if i_d * half_domains + j_d == target_index else "Dropped"
                            ax.text(0.5, 0.5, text_to_show, horizontalalignment="center",
                                    verticalalignment="center", transform=ax.transAxes, fontsize=40)
                            ax.imshow(get_checker_image(), interpolation="nearest", cmap="gray", vmin=0, vmax=1)
                        else:
                            # a real image to show
                            ax.imshow(image, interpolation="nearest")
                        ax.axis("off")

            # add row titles to the first column, if available
            if row_titles is not None and j == 0:
                sub_figs[i, j].text(0, 0.5, row_titles[i], horizontalalignment="center",
                                    verticalalignment="center", rotation=90, fontsize=40)

    fig.patch.set_alpha(0.0)
    fig.tight_layout()
    return fig


def show_single_input_model_comparison(source_images, target_images, genned_images, title="", model_names=None):
    batch_shape = tf.shape(source_images)
    batch_size = batch_shape[0].numpy()
    cols = len(genned_images) + 2
    rows = batch_size

    if model_names is None:
        model_names = ["Pix2Pix", "StarGAN", "CollaGAN"]
    source_images = source_images * 0.5 + 0.5
    target_images = target_images * 0.5 + 0.5
    genned_images = [genned_images[i] * 0.5 + 0.5 for i in range(len(genned_images))]

    column_titles = ["Source", "Target", *model_names]
    column_contents = [source_images, target_images, *genned_images]

    fig = plt.figure(figsize=(8 * cols, 8 * rows))
    plt.suptitle(title)
    for i in range(rows):
        for j in range(cols):
            index = i * cols + j + 1
            plt.subplot(rows, cols, index)
            plt.title(column_titles[j] if i == 0 else "", fontsize=48)
            plt.imshow(column_contents[j][i], interpolation="nearest")
            plt.axis("off")
    fig.patch.set_alpha(0.0)
    plt.tight_layout()
    plt.show()


def show_single_input_model_matrix(images, model_name):
    rows = cols = len(DOMAINS)
    f = plt.figure(figsize=(2 * cols, 2 * rows))
    plt.suptitle(model_name)
    for i in range(rows):
        for j in range(cols):
            image = images[i][j] * 0.5 + 0.5
            plt.subplot(rows, cols, i * cols + j + 1)
            plt.title(f"{DOMAINS[j]} -> {DOMAINS[i]}")
            plt.imshow(image, interpolation="nearest")
            plt.axis("off")
    return f


def get_checker_image():
    checkers = tf.tile(tf.constant([[0, 1], [1, 0]], dtype=tf.float32), [4, 4])
    rgba_checkers = tf.stack([checkers, checkers, checkers, tf.ones_like(checkers)], axis=-1) * 0.2 + 0.8
    return tf.pad(rgba_checkers, [[1, 1], [1, 1], [0, 0]], constant_values=0)


def get_transparent_image():
    return tf.tile(tf.constant([[[0, 0, 0, 0]]], dtype=tf.float32), [64, 64, 1])


def get_empty_image_text(col_idx, target):
    if col_idx == 0:
        return [("Target", 0.5, 0.5, "center", "center")]
    else:
        return []


def show_multiple_input_model_comparison(source_images, target_images, genned_images, title="", model_names=[],
                                         target_indices=None):
    # Generates the RIGHT images from the other 3 inputs
    batch_shape = tf.shape(target_images)
    batch_size = batch_shape[0].numpy()
    if target_indices is None:
        target_indices = [c // 4 for c in range(batch_size)]

    target_images = target_images * 0.5 + 0.5
    genned_images = [genned_images[i] * 0.5 + 0.5 for i in range(len(genned_images))]

    half_domains = len(DOMAINS) // 2
    column_titles = ["Source", "Target", *model_names[:len(genned_images)]]
    column_contents = [source_images, target_images, *genned_images]
    cols = len(column_titles)
    rows = batch_size

    fig = plt.figure(figsize=(8 * cols, 8 * rows))
    plt.suptitle(title)
    is_subplot_column = lambda j: j == 0
    is_target_image_subsubplot = lambda i_d, j_d, idx: 2 * i_d + j_d == idx
    sub_figs = fig.subfigures(rows, cols)
    for i in range(rows):
        for j in range(cols):
            sub_figs[i, j].suptitle(column_titles[j] if i == 0 else "", fontsize=48)
            sub_figs[i, j].patch.set_alpha(0.0)

            if is_subplot_column(j):
                axes = sub_figs[i, j].subplots(half_domains, half_domains)
                for i_d in range(half_domains):
                    for j_d in range(half_domains):
                        ax = axes[i_d, j_d]
                        image = column_contents[j][i][i_d * half_domains + j_d]
                        if tf.math.count_nonzero(image) == 0:
                            # no image (either the target or a dropped image in the sources column)
                            target_index = target_indices[i]
                            text_to_show = "Target" if is_target_image_subsubplot(i_d, j_d, target_index) else "Dropped"
                            ax.text(0.5, 0.5, text_to_show, horizontalalignment="center",
                                    verticalalignment="center", transform=ax.transAxes, fontsize=40)
                            ax.imshow(get_checker_image(), interpolation="nearest", cmap="gray", vmin=0, vmax=1)
                        else:
                            # a real image to show
                            ax.imshow(image * 0.5 + 0.5, interpolation="nearest")
                        ax.axis("off")
            else:
                ax = sub_figs[i, j].subplots(1, 1)
                ax.imshow(column_contents[j][i], interpolation="nearest")
                ax.axis("off")

    fig.patch.set_alpha(0.0)
    # plt.tight_layout()
    # plt.show()
    return fig


def show_collagan3_and_baseline_comparison(source_images, target_images, genned_images, title="",
                                           model_names=None):
    # source_images is (b, d, s, s, c)
    # target_images is (b, s, s, c)
    # genned_images is [(b, d, s, s, c), (b, d, s, s, c), (b, s, s, c)]
    if model_names is None:
        model_names = ["Pix2Pix", "StarGAN", "CollaGAN-3", "C3 Post Processed"]
    batch_shape = tf.shape(source_images)
    batch_size = batch_shape[0].numpy()
    is_subplot_column = lambda col_idx: col_idx in [0, 2, 3]  # source, p2p, star
    get_empty_image_replacement = lambda col_idx: get_checker_image() if col_idx == 0 else get_transparent_image()
    half_domains = len(DOMAINS) // 2

    source_images = source_images * 0.5 + 0.5
    target_images = target_images * 0.5 + 0.5
    genned_images = [genned_images[i] * 0.5 + 0.5 for i in range(len(genned_images))]
    target_indices = []
    for i in range(batch_size):
        pixel_sum = tf.reduce_sum(source_images[i] * 2.0 - 1.0, axis=[1, 2, 3])
        target_index = tf.argmin(pixel_sum)
        target_indices.append(target_index)

    column_titles = ["Source", "Target", *model_names]
    column_contents = [source_images, target_images, *genned_images]

    cols = len(genned_images) + 2
    rows = batch_size

    fig = plt.figure(figsize=(8 * cols, 8 * rows))
    plt.suptitle(title, fontsize=40)
    sub_figs = fig.subfigures(rows, cols)
    for i in range(rows):
        for j in range(cols):
            sub_figs[i, j].suptitle(column_titles[j] if i == 0 else "", fontsize=40)
            sub_figs[i, j].patch.set_alpha(0.0)

            if is_subplot_column(j):
                if j == 0 or column_titles[j] != "CollaGAN-2":
                    # get a 2x2 subplots axes
                    axes = sub_figs[i, j].subplots(half_domains, half_domains)
                else:
                    # get a 1 then 2 subplots axes (for CollaGAN-2)
                    axes = []

                    middle_at_top = [0.325, 0.525, 0.35, 0.35]
                    left_at_top = [0.125, 0.525, 0.35, 0.35]
                    right_at_top = [0.55, 0.525, 0.35, 0.35]
                    middle_at_bot = [0.325, 0.125, 0.35, 0.35]
                    left_at_bot = [0.125, 0.125, 0.35, 0.35]
                    right_at_bot = [0.55, 0.125, 0.35, 0.35]
                    full = [0, 0, 1, 1]
                    target_index = target_indices[i]
                    if target_index == 0:
                        axes = [full, middle_at_top, left_at_bot, right_at_bot]
                    elif target_index == 1:
                        axes = [middle_at_top, full, left_at_bot, right_at_bot]
                    elif target_index == 2:
                        axes = [left_at_top, right_at_top, full, middle_at_bot]
                    elif target_index == 3:
                        axes = [left_at_top, right_at_top, middle_at_bot, full]
                    axes = [sub_figs[i, j].add_axes(rect) for rect in axes]

                for i_d in range(half_domains):
                    for j_d in range(half_domains):
                        source_index = i_d * 2 + j_d
                        ax = axes[i_d, j_d] if isinstance(axes, np.ndarray) else axes[source_index]
                        image = column_contents[j][i][i_d * 2 + j_d]
                        image_params = {}
                        if tf.math.count_nonzero(image) == 0:
                            # no image (either the target image in the sources column, or the target spot in the
                            # generated images)
                            target_index = i_d * 2 + j_d
                            image = get_empty_image_replacement(j)
                            texts = get_empty_image_text(j, target_index)
                            image_params = {"cmap": "gray", "vmin": 0, "vmax": 1}
                            for t, x, y, ha, va in texts:
                                ax.text(x, y, t, horizontalalignment=ha,
                                        verticalalignment=va, transform=ax.transAxes, fontsize=40)
                        else:
                            # a real image to show
                            if column_titles[j] in ["Pix2Pix", "StarGAN", "CollaGAN-1"]:
                                ax.set_title(f"From {DOMAINS[source_index]}", fontsize=32, y=1, pad=-8)
                            elif column_titles[j] == "CollaGAN-2":
                                additional_non_source = i_d * 2 + j_d
                                target_index = target_indices[i]
                                source_domains = [n for n in range(4) if
                                                  n != target_index and n != additional_non_source]
                                ax.set_title(f"From {DOMAINS[source_domains[0]][0]}+{DOMAINS[source_domains[1]][0]}",
                                             fontsize=32, y=1, pad=-8)
                        ax.imshow(image, interpolation="nearest", **image_params)
                        ax.axis("off")
            else:
                ax = sub_figs[i, j].subplots(1, 1)
                ax.imshow(column_contents[j][i], interpolation="nearest")
                ax.axis("off")

    return fig


def show_collagan_input_comparison(source_images, target_images, genned_images, title="", model_names=None):
    if model_names is None:
        model_names = ["CollaGAN-1", "CollaGAN-2", "CollaGAN-3"]
    return show_collagan3_and_baseline_comparison(source_images, target_images, genned_images, title, model_names)


def show_single_input_comparison(source_images, target_images, genned_images, title=""):
    # source_images is (b, d, s, s, c)
    # target_images is (b, s, s, c)
    # genned_images is [(b, d, s, s, c), (b, d, s, s, c), (b, s, s, c)]
    batch_shape = tf.shape(source_images)
    batch_size = batch_shape[0].numpy()
    model_names = ["Pix2Pix", "StarGAN", "CollaGAN-1"]
    is_subplot_column = lambda col_idx: col_idx in [0, 2, 3, 4]  # source, p2p, star, colla1
    get_empty_image_replacement = lambda col_idx: get_checker_image() if col_idx == 0 else get_transparent_image()
    half_domains = len(DOMAINS) // 2

    source_images = source_images * 0.5 + 0.5
    target_images = target_images * 0.5 + 0.5
    genned_images = [genned_images[i] * 0.5 + 0.5 for i in range(len(genned_images))]

    column_titles = ["Source", "Target", *model_names]
    column_contents = [source_images, target_images, *genned_images]

    cols = len(genned_images) + 2
    rows = batch_size

    fig = plt.figure(figsize=(8 * cols, 8 * rows))
    plt.suptitle(title, fontsize=40)
    sub_figs = fig.subfigures(rows, cols)
    for i in range(rows):
        for j in range(cols):
            sub_figs[i, j].suptitle(column_titles[j] if i == 0 else "", fontsize=40)
            sub_figs[i, j].patch.set_alpha(0.0)

            if is_subplot_column(j):
                axes = sub_figs[i, j].subplots(half_domains, half_domains)
                for i_d in range(half_domains):
                    for j_d in range(half_domains):
                        source_index = i_d * 2 + j_d
                        ax = axes[i_d, j_d]
                        image = column_contents[j][i][i_d * 2 + j_d]
                        image_params = {}
                        if tf.math.count_nonzero(image) == 0:
                            # no image (either the target image in the sources column, or the target spot in the
                            # generated images)
                            target_index = i_d * 2 + j_d
                            image = get_empty_image_replacement(j)
                            texts = get_empty_image_text(j, target_index)
                            image_params = {"cmap": "gray", "vmin": 0, "vmax": 1}
                            for t, x, y, ha, va in texts:
                                ax.text(x, y, t, horizontalalignment=ha,
                                        verticalalignment=va, transform=ax.transAxes, fontsize=40)
                        else:
                            # a real image to show
                            if j != 0:
                                ax.set_title(f"From {DOMAINS[source_index]}", fontsize=32, y=1, pad=-8)
                        ax.imshow(image, interpolation="nearest", **image_params)
                        ax.axis("off")
            else:
                ax = sub_figs[i, j].subplots(1, 1)
                ax.imshow(column_contents[j][i], interpolation="nearest")
                ax.axis("off")

    return fig
