import tensorflow as tf
from skimage.measure import label, regionprops

def split_eyeside(binary_cup_mask:tf.Tensor, binary_disc_mask:tf.Tensor, dataset_img:tf.Tensor):
    result = []

    # loop over the batch
    for i in range(binary_cup_mask.shape[0]):
        cup = binary_cup_mask[i].numpy()
        disc = binary_disc_mask[i].numpy()
        img = dataset_img[i].numpy()

        # --- bounding boxes ---
        cup_props = regionprops(label(cup))
        disc_props = regionprops(label(disc))

        if len(cup_props) == 0 or len(disc_props) == 0:
            result.append({
                "eye_side": "uncertain",
                "left_intensity": None,
                "right_intensity": None,
                "cup_bbox": None,
                "disc_bbox": None,
            })
            continue

        cup_bbox = cup_props[0].bbox
        disc_bbox = disc_props[0].bbox

        # crop disc area
        disc_crop = img[disc_bbox[0]:disc_bbox[2], disc_bbox[1]:disc_bbox[3], :]

        if disc_crop.size == 0: # invalid crop
            result.append({
                "eye_side": "uncertain",
                "left_intensity": None,
                "right_intensity": None,
                "cup_bbox": cup_bbox,
                "disc_bbox": disc_bbox,
            })
            continue

        # split disc crop
        h, w, _ = disc_crop.shape
        mid = w // 2
        left = disc_crop[:, :mid, :]
        right = disc_crop[:, mid:, :]

        # green channel
        left_green = left[..., 1]
        right_green = right[..., 1]

        # intensities
        left_intensity = float(tf.reduce_sum(left_green).numpy())
        right_intensity = float(tf.reduce_sum(right_green).numpy())

        # decision of eye side
        if left_intensity > right_intensity:
            eye_side = "right"
        elif left_intensity < right_intensity:
            eye_side = "left"
        else:
            eye_side = "uncertain"

        # collect result
        result.append({
            "eye_side": eye_side,
            "left_intensity": left_intensity,
            "right_intensity": right_intensity,
            "cup_bbox": cup_bbox,
            "disc_bbox": disc_bbox,
        })

    return result