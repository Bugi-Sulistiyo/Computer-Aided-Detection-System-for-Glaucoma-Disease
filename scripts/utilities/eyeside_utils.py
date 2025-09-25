import tensorflow as tf
from skimage.measure import label, regionprops

def split_eyeside(binary_cup_mask:tf.Tensor, binary_disc_mask:tf.Tensor, dataset_img:tf.Tensor, img_paths:tf.Tensor):
    def largest_region(props):
        return max(props, key=lambda p: p.area) if props else None
    
    result = []
    side_map = {"l": "left", "r": "right"}

    # loop over the batch
    for index in range(binary_cup_mask.shape[0]):
        # filenames & labels
        path_str = img_paths[index].numpy().decode("utf-8")
        file_name = path_str.split("\\")[-1]
        real_label = {"l": "left", "r": "right"}.get(file_name.split("_")[3], "unknown")
        # mask & images
        cup = binary_cup_mask[index].numpy()
        disc = binary_disc_mask[index].numpy()
        img = dataset_img[index].numpy()
        # base entry
        entry = {
            "file_name": file_name,
            "real_eye_side": real_label,
            "pred_eye_side": "uncertain",
            "left_intensity": None,
            "right_intensity": None,
            "cup_bbox": None,
            "disc_bbox": None,
        }

        # --- bounding boxes ---
        cup_props = largest_region(regionprops(label(cup)))
        disc_props = largest_region(regionprops(label(disc)))

        if not cup_props or not disc_props:
            result.append(entry)
            continue

        cup_bbox = cup_props.bbox
        disc_bbox = disc_props.bbox
        entry.update({
            "cup_bbox": cup_bbox,
            "disc_bbox": disc_bbox,
        })

        # crop disc area
        disc_crop = img[disc_bbox[0]:disc_bbox[2], disc_bbox[1]:disc_bbox[3], :]

        if disc_crop.size == 0: # invalid crop
            result.append(entry)
            continue

        # split disc into left and right
        h, w, _ = disc_crop.shape
        mid = w // 2
        # get only the green channel
        left_green = disc_crop[:, :mid, 1]
        right_green = disc_crop[:, mid:, 1]

        # intensities
        left_intensity = float(left_green.sum())
        right_intensity = float(right_green.sum())

        # decision of eye side
        if left_intensity > right_intensity:
            eye_side = "right"
        elif left_intensity < right_intensity:
            eye_side = "left"
        else:
            eye_side = "uncertain"

        # collect result
        entry.update({
            "pred_eye_side": eye_side,
            "left_intensity": left_intensity,
            "right_intensity": right_intensity,
        })
        result.append(entry)

    return result