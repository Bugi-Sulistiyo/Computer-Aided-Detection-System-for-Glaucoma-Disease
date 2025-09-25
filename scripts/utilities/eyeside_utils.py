import tensorflow as tf
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt

def largest_region(mask):
    props = regionprops(label(mask))
    return max(props, key=lambda p: p.area) if props else None

def split_eyeside(dataset:tf.data.Dataset, model:tf.keras.Model, treshold:float=.5):
    
    result = []
    side_map = {"l": "left", "r": "right"}
    idx_across_batches = 0 # global index across batches

    for images, _, img_paths in dataset:
        # predick a mask image
        pred_mask = model.predict(images, verbose=0)
        batch_size = images.shape[0]

        # loop over the batch
        for index in range(batch_size):
            # filenames & labels
            path_str = img_paths[index].numpy().decode("utf-8")
            file_name = path_str.split("\\")[-1]
            real_label = side_map.get(file_name.split("_")[3], "unknown")
            # mask & images
            cup_mask = (pred_mask[index, ..., 1] > treshold).astype("uint8")
            disc_mask = (pred_mask[index, ..., 2] > treshold).astype("uint8")
            img = images[index].numpy()

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
            cup_props = largest_region(cup_mask)
            disc_props = largest_region(disc_mask)

            if not cup_props or not disc_props:
                result.append(entry)
                idx_across_batches += 1
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
                idx_across_batches += 1
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
            idx_across_batches += 1

    return result

def get_img(path:str, img_size:int):
    img = tf.io.read_file(path)                                         # read the image
    img = tf.image.decode_jpeg(img, channels=3)                             # read the jpg file to readable type
    img = tf.image.resize(img, (img_size, img_size), method='nearest')      # change the image size
    img = tf.cast(img, tf.float32) / 255.                                   # normalize the image
    return tf.expand_dims(img, axis=0)                                       # change the shape of the image to simulate batch dataset

def visualize_result_eye_side(model:tf.keras.Model,
                                img_path:str, mask_path:str,
                                threshold:float = .5, img_size:int=128,
                                visualize:bool=True):
    # preprocess and get the image
    img = get_img(img_path, img_size)
    # predict the image
    pred_mask = model.predict(img, verbose=0)

    # split the mask image to a binary mask image
    cup_mask = (pred_mask[..., 1] > threshold).astype("uint8")[0]
    disc_mask = (pred_mask[..., 2] > threshold).astype("uint8")[0]

    # create a bounding box for each binary image (will get the coordinate)
    cup_prop = largest_region(cup_mask)
    disc_prop = largest_region(disc_mask)

    if not cup_prop or not disc_prop:
        return {
            "file_name": img_path.split("\\")[-1],
            "pred_eye_side": "uncertain",
            "left_intensity": None,
            "right_intensity": None,
            "cup_bbox": None,
            "disc_bbox": None,
        }
    
    cup_bbox = cup_prop.bbox
    disc_bbox = disc_prop.bbox

    # --- crop disc area ---
    disc_crop = img[0, disc_bbox[0]:disc_bbox[2], disc_bbox[1]:disc_bbox[3], :].numpy()
    h, w, _ = disc_crop.shape
    mid = w // 2

    # --- split & intensities ---
    left_intensity = disc_crop[:, :mid, 1].sum()
    right_intensity = disc_crop[:, mid:, 1].sum()

    # --- decide eye side ---
    if left_intensity > right_intensity:
        eye_side = "right"
    elif left_intensity < right_intensity:
        eye_side = "left"
    else:
        eye_side = "uncertain"
    
    # --- visualization ---
    if visualize:
        fig, axes = plt.subplots(3, 2, figsize=(10, 10))

        axes[0, 0].imshow(plt.imread(mask_path), cmap="gray")
        axes[0, 0].set_title("Original Mask")
        axes[0, 0].axis("off")

        axes[0, 1].imshow(tf.argmax(pred_mask, axis=-1)[0], cmap="gray")
        axes[0, 1].add_patch(plt.Rectangle((cup_bbox[1], cup_bbox[0]),
                                            cup_bbox[3]-cup_bbox[1],
                                            cup_bbox[2]-cup_bbox[0],
                                            edgecolor='b', facecolor='none'))
        axes[0, 1].add_patch(plt.Rectangle((disc_bbox[1], disc_bbox[0]),
                                            disc_bbox[3]-disc_bbox[1],
                                            disc_bbox[2]-disc_bbox[0],
                                            edgecolor='c', facecolor='none'))
        axes[0, 1].set_title("Predicted Mask")
        axes[0, 1].axis("off")

        axes[1, 0].imshow(img[0])
        axes[1, 0].add_patch(plt.Rectangle((cup_bbox[1], cup_bbox[0]),
                                            cup_bbox[3]-cup_bbox[1],
                                            cup_bbox[2]-cup_bbox[0],
                                            edgecolor='b', facecolor='none'))
        axes[1, 0].add_patch(plt.Rectangle((disc_bbox[1], disc_bbox[0]),
                                            disc_bbox[3]-disc_bbox[1],
                                            disc_bbox[2]-disc_bbox[0],
                                            edgecolor='c', facecolor='none'))
        axes[1, 0].set_title("Original Image")
        axes[1, 0].axis("off")

        axes[1, 1].imshow(disc_crop)
        axes[1, 1].axvline(x=mid, color="g", linestyle="--", linewidth=2)
        axes[1, 1].set_title(f"Cropped Disc Area ({eye_side} eye)")
        axes[1, 1].axis("off")

        axes[2, 0].imshow(disc_crop[:, :mid, 1], cmap="gray")
        axes[2, 0].set_title(f"Left Green Channel: {left_intensity:.2f}")
        axes[2, 0].axis("off")

        axes[2, 1].imshow(disc_crop[:, mid:, 1], cmap="gray")
        axes[2, 1].set_title(f"Right Green Channel: {right_intensity:.2f}")
        axes[2, 1].axis("off")

        plt.tight_layout()
        plt.show()

    # --- results ---
    return {
        "file_name": img_path.split("\\")[-1],
        "pred_eye_side": eye_side,
        "left_intensity": float(left_intensity),
        "right_intensity": float(right_intensity),
        "cup_bbox": cup_bbox,
        "disc_bbox": disc_bbox,
    }