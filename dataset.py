# dataset.py
"""
Dataset preparation, augmentation, and sample generation functions.
"""
import numpy as np
import cv2
import random
import torch
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter
import os
import glob
import matplotlib.pyplot as plt

from config import (AUGMENTATION_SIZE, MODEL_INPUT_SIZE, MIN_DIM_RESCALE, GT_PSF_SIGMA, IMAGE_DIR_TRAIN_VAL, GT_DIR_TRAIN_VAL, DEVICE, TARGET_POINT_SELECTION_SKEW_POWER) # Added TARGET_POINT_SELECTION_SKEW_POWER

IMG_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMG_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

def prepare_data_augmentations(image, gt_coor, target_size=AUGMENTATION_SIZE, min_dim=MIN_DIM_RESCALE, crop_region_factor=1.0):
    """
    Prepares augmented image and ground truth coordinates.
    Args:
        image (np.ndarray): Input image.
        gt_coor (np.ndarray): Ground truth coordinates.
        target_size (int): The size (height and width) of the augmented output crop.
        min_dim (int): Minimum dimension after rescaling.
        crop_region_factor (float): Factor (0.0 to 1.0) to control the vertical region for random cropping.
                                   1.0 means full range, 0.5 means crop must start in the top half of possible y-locations.
    """
    if image is None:
        print("Warning: prepare_data_augmentations received None image.")
        return None, None
    h, w, c = image.shape
    if c != 3:
        print(f"Warning: Expected a 3-channel image (H, W, 3), but got {c} channels. Attempting to proceed.")

    scale_factor = random.uniform(0.7, 1.3)
    new_h, new_w = int(h * scale_factor), int(w * scale_factor)
    if min(h, w) > 0 and min(new_h, new_w) < min_dim:
        scale_factor = min_dim / min(h, w)
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
    elif min(h, w) == 0:
        print("Warning: Image has zero dimension before scaling.")
        return None, None
    new_h = max(1, new_h)
    new_w = max(1, new_w)
    try:
        scaled_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    except cv2.error as e:
        print(f"Error during cv2.resize: {e}. Original shape: {(h,w)}, Target shape: {(new_h, new_w)}")
        return None, None
    scaled_gt_coor = gt_coor * scale_factor if gt_coor is not None and gt_coor.size > 0 else np.array([])

    curr_h, curr_w = scaled_image.shape[:2]
    # crop_h = min(target_size, curr_h) # Target size is used for the actual crop dimensions later
    # crop_w = min(target_size, curr_w)
    cropped_gt_coor = np.array([])

    if curr_h >= target_size and curr_w >= target_size:
        max_x_start_offset = curr_w - target_size
        max_y_start_offset = curr_h - target_size

        effective_max_y_start = int(max_y_start_offset * crop_region_factor)
        effective_max_y_start = max(0, effective_max_y_start) 

        x_min = random.randint(0, max_x_start_offset)
        y_min = random.randint(0, effective_max_y_start)
        
        cropped_image = scaled_image[y_min : y_min + target_size, x_min : x_min + target_size]
        if scaled_gt_coor.size > 0:
            keep_mask = (scaled_gt_coor[:, 0] >= x_min) & (scaled_gt_coor[:, 0] < x_min + target_size) & \
                        (scaled_gt_coor[:, 1] >= y_min) & (scaled_gt_coor[:, 1] < y_min + target_size)
            cropped_gt_coor = scaled_gt_coor[keep_mask]
            if cropped_gt_coor.size > 0:
                cropped_gt_coor[:, 0] -= x_min
                cropped_gt_coor[:, 1] -= y_min
        final_h, final_w = target_size, target_size
    else:
        print(f"Warning: Scaled image ({curr_h}x{curr_w}) smaller than target ({target_size}x{target_size}). Resizing up.")
        try:
             cropped_image = cv2.resize(scaled_image, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        except cv2.error as e:
             print(f"Error during cv2.resize (upscaling): {e}. Scaled shape: {(curr_h, curr_w)}, Target shape: {(target_size, target_size)}")
             return None, None
        if curr_w > 0 and curr_h > 0 and scaled_gt_coor.size > 0:
            scale_x_final = target_size / float(curr_w)
            scale_y_final = target_size / float(curr_h)
            cropped_gt_coor = scaled_gt_coor.copy()
            cropped_gt_coor[:, 0] *= scale_x_final
            cropped_gt_coor[:, 1] *= scale_y_final
            cropped_gt_coor[:, 0] = np.clip(cropped_gt_coor[:, 0], 0, target_size - 1)
            cropped_gt_coor[:, 1] = np.clip(cropped_gt_coor[:, 1], 0, target_size - 1)
        final_h, final_w = target_size, target_size

    if random.random() < 0.5:
        cropped_image = cv2.flip(cropped_image, 1)
        if cropped_gt_coor.size > 0:
            cropped_gt_coor[:, 0] = (final_w - 1) - cropped_gt_coor[:, 0]

    if cropped_image.shape[0] != target_size or cropped_image.shape[1] != target_size:
         print(f"Warning: Final augmented image shape {cropped_image.shape} does not match target {target_size}x{target_size}. Resizing again.")
         try:
            cropped_image = cv2.resize(cropped_image, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
         except cv2.error as e:
            print(f"Error during final cv2.resize: {e}. Shape before: {cropped_image.shape}, Target shape: {(target_size, target_size)}")
            return None, None
    return cropped_image, cropped_gt_coor

def generate_single_psf(coord, image_shape, sigma):
    height, width = image_shape
    psf = np.zeros((height, width), dtype=np.float32)
    x = np.clip(int(round(coord[0])), 0, width - 1)
    y = np.clip(int(round(coord[1])), 0, height - 1)
    psf[y, x] = 1.0
    psf = gaussian_filter(psf, sigma=sigma, order=0, mode='constant', cval=0.0)
    psf_sum = np.sum(psf)
    if psf_sum > 1e-7:
        psf /= psf_sum
    return psf

def get_center_crop_coords(image_size, crop_size):
    img_h, img_w = image_size
    crop_h, crop_w = crop_size
    start_y = max(0, (img_h - crop_h) // 2)
    start_x = max(0, (img_w - crop_w) // 2)
    return start_y, start_x

def get_random_coord_index_in_center_skewed(coordinates, image_shape, center_crop_shape, skew_power=0.0, epsilon=1e-6):
    """
    Selects a random coordinate index from 'coordinates' that falls within the center crop.
    The selection can be skewed based on the y-coordinate.
    Args:
        coordinates (np.ndarray): Array of (x,y) points, assumed to be sorted as desired for timesteps.
        image_shape (tuple): (H, W) of the image space these coordinates belong to (augmented image).
        center_crop_shape (tuple): (crop_H, crop_W) of the final model input size.
        skew_power (float): Power for weighting. 0 for uniform. >0 biases to smaller y-values (higher in image).
        epsilon (float): Small constant for numerical stability in weighting.
    Returns:
        int or None: The index (from the input `coordinates` array) of the selected point, or None.
    """
    if coordinates is None or coordinates.shape[0] == 0:
        return None

    aug_img_h, _ = image_shape # Height of the augmented image before center crop
    crop_h, crop_w = center_crop_shape
    start_y_crop, start_x_crop = get_center_crop_coords(image_shape, center_crop_shape)
    end_y_crop, end_x_crop = start_y_crop + crop_h, start_x_crop + crop_w

    candidate_indices_in_input_coords = []
    candidate_y_coords_in_aug_image = [] # Y-coordinates relative to the augmented image

    for i, (x, y) in enumerate(coordinates):
        # x, y are coordinates in the augmented image space
        if start_x_crop <= x < end_x_crop and start_y_crop <= y < end_y_crop:
            candidate_indices_in_input_coords.append(i)
            candidate_y_coords_in_aug_image.append(y)
    
    if not candidate_indices_in_input_coords:
        return None

    if skew_power == 0.0 or len(candidate_indices_in_input_coords) == 1:
        # Uniform selection or only one choice
        return random.choice(candidate_indices_in_input_coords)
    else:
        weights = []
        # Use y-coords from the augmented image for weighting
        # Smaller y means higher in the image, thus "further back"
        # To give higher weight to smaller y: (max_y_candidate - y + epsilon) or (1 / (y + epsilon))
        
        # Let's use a simpler approach: directly weight by how "high up" a point is.
        # Higher y values in image coordinates are lower on the screen.
        # We want to bias towards smaller y values (higher on screen).
        # Weights: (aug_img_h - y + epsilon) ** skew_power
        # Or, relative to candidates: (max_candidate_y - y + epsilon) ** skew_power
        
        # Using y-coordinates relative to the candidates present in the center crop
        # This makes the skew relative to the current selection pool.
        max_y_among_candidates = np.max(candidate_y_coords_in_aug_image)

        for y_coord in candidate_y_coords_in_aug_image:
            # Weight: (max_y_among_candidates - y_coord + epsilon)^skew_power
            # This gives higher weight to smaller y_coord values.
            weight = (max_y_among_candidates - y_coord + epsilon) ** skew_power
            weights.append(weight)
        
        # Normalize weights (random.choices does this, but good for inspection)
        sum_weights = sum(weights)
        if sum_weights < 1e-9: # All weights effectively zero (e.g. all y_coords are max_y)
             return random.choice(candidate_indices_in_input_coords) # Fallback to uniform

        chosen_index_from_candidates = random.choices(
            list(range(len(candidate_indices_in_input_coords))), 
            weights=weights, 
            k=1
        )[0]
        
        return candidate_indices_in_input_coords[chosen_index_from_candidates]


def generate_train_sample(image_paths, gt_paths, augment_size=AUGMENTATION_SIZE,
                          model_input_size=MODEL_INPUT_SIZE, psf_sigma=GT_PSF_SIGMA,
                          negative_prob=0.1, current_curriculum_crop_region_factor=1.0,
                          target_point_skew_power=0.0): # Added target_point_skew_power
    max_retries = 10
    for _ in range(max_retries):
        rand_idx = random.randint(0, len(image_paths) - 1)
        image_path = image_paths[rand_idx]
        img_filename = os.path.basename(image_path)
        gt_filename = "GT_" + os.path.splitext(img_filename)[0] + ".mat"
        # Corrected GT path construction: use directory from gt_paths[0] or relative to image
        if gt_paths: # If gt_paths is provided and not empty
            gt_base_dir = os.path.dirname(gt_paths[0])
            gt_path = os.path.join(gt_base_dir, gt_filename)
        else: # Fallback if gt_paths is not useful (e.g. empty or not matching structure)
            image_dir = os.path.dirname(image_path)
            # Attempt to find ground_truth relative to images folder
            # This assumes a structure like .../ShanghaiTech/part_A_final/train_data/images and .../ShanghaiTech/part_A_final/train_data/ground_truth
            gt_dir_relative = os.path.join(os.path.dirname(image_dir), "ground_truth")
            if not os.path.exists(gt_dir_relative): # Try one level higher if "train_data" is a subdir
                 gt_dir_relative = os.path.join(os.path.dirname(os.path.dirname(image_dir)), "ground_truth")

            gt_path = os.path.join(gt_dir_relative, gt_filename)


        if not os.path.exists(gt_path):
             # Fallback to using the corresponding index from gt_paths if it exists and the constructed one failed
             if rand_idx < len(gt_paths) and os.path.exists(gt_paths[rand_idx]):
                 gt_path = gt_paths[rand_idx]
             else:
                 print(f"Warning: GT file not found for {image_path} (tried {gt_path}). Skipping.")
                 continue
        
        image = cv2.imread(image_path)
        if image is None: print(f"Warning: Failed to load image {image_path}. Skipping."); continue
        if len(image.shape) == 2: image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4: image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        elif image.shape[2] != 3: print(f"Warning: Image {image_path} has unexpected shape {image.shape}. Skipping."); continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        try:
            mat_data = loadmat(gt_path)
            if 'image_info' in mat_data: gt_coor = mat_data['image_info'][0, 0][0, 0][0].astype(np.float32)
            elif 'annPoints' in mat_data: gt_coor = mat_data['annPoints'].astype(np.float32)
            else:
                 found_coords = False
                 for key, value in mat_data.items():
                     if isinstance(value, np.ndarray) and len(value.shape) == 2 and value.shape[1] == 2:
                         gt_coor = value.astype(np.float32); found_coords = True; break
                 if not found_coords: print(f"Warning: Could not find coordinate data in {gt_path}. Skipping."); continue
        except Exception as e: print(f"Warning: Error loading/parsing .mat {gt_path}: {e}. Skipping."); continue

        if gt_coor.shape[0] == 0: continue 

        aug_image, aug_gt_coor = prepare_data_augmentations(
            image, gt_coor.copy(), 
            target_size=augment_size, 
            crop_region_factor=current_curriculum_crop_region_factor
        )
        if aug_image is None or aug_gt_coor is None: continue
        
        if aug_gt_coor.shape[0] < 1: continue 

        img_h, img_w = aug_image.shape[:2] # These are dimensions of aug_image (e.g. 256x256)
        if img_h == 0 or img_w == 0: print(f"Warning: Augmented image has zero dimension for {image_path}. Skipping."); continue

        sorted_indices = np.lexsort((aug_gt_coor[:, 0], -aug_gt_coor[:, 1]))
        sorted_aug_gt_coor = aug_gt_coor[sorted_indices]
        num_actual_points_in_aug_crop = len(sorted_aug_gt_coor)

        is_negative_sample = random.random() < negative_prob
        num_previous_points_for_input_psf = 0

        if is_negative_sample:
            confidence_target = 0.0
            target_psf_full = np.zeros((img_h, img_w), dtype=np.float32) 
            num_previous_points_for_input_psf = num_actual_points_in_aug_crop
        else: 
            confidence_target = 1.0
            center_crop_shape_for_model = (model_input_size, model_input_size)
            
            # Use the new skewed selection function
            # sorted_aug_gt_coor are points in the augmented image space (e.g., 256x256)
            # image_shape for get_random_coord_index_in_center_skewed should be (img_h, img_w) of aug_image
            timestep_k_for_target = get_random_coord_index_in_center_skewed(
                coordinates=sorted_aug_gt_coor, 
                image_shape=(img_h, img_w), # Shape of aug_image
                center_crop_shape=center_crop_shape_for_model, 
                skew_power=target_point_skew_power
            )
            
            if timestep_k_for_target is None: 
                 continue

            target_psf_full = generate_single_psf(sorted_aug_gt_coor[timestep_k_for_target], (img_h, img_w), psf_sigma)
            num_previous_points_for_input_psf = timestep_k_for_target

        input_psf_full = np.zeros((img_h, img_w), dtype=np.float32)
        if num_previous_points_for_input_psf > 0:
            previous_points_map = np.zeros((img_h, img_w), dtype=np.float32)
            for i in range(num_previous_points_for_input_psf):
                coord = sorted_aug_gt_coor[i]
                x = np.clip(int(round(coord[0])), 0, img_w - 1)
                y = np.clip(int(round(coord[1])), 0, img_h - 1)
                previous_points_map[y, x] += 1.0
            if np.sum(previous_points_map) > 1e-7:
                 input_psf_full = gaussian_filter(previous_points_map, sigma=psf_sigma, order=0, mode='constant', cval=0.0)

        center_crop_shape_for_model = (model_input_size, model_input_size)
        start_y, start_x = get_center_crop_coords((img_h, img_w), center_crop_shape_for_model) # Crop from aug_image
        end_y, end_x = start_y + model_input_size, start_x + model_input_size
        if start_y < 0 or start_x < 0 or end_y > img_h or end_x > img_w: print(f"Warning: Invalid crop coordinates for {image_path}. Skipping."); continue

        final_image = aug_image[start_y:end_y, start_x:end_x]
        final_input_psf = input_psf_full[start_y:end_y, start_x:end_x]
        final_target_psf = target_psf_full[start_y:end_y, start_x:end_x]

        if final_image.shape[:2] != center_crop_shape_for_model or \
           final_input_psf.shape != center_crop_shape_for_model or \
           final_target_psf.shape != center_crop_shape_for_model: print(f"Warning: Cropped shape mismatch for {image_path}. Skipping."); continue

        final_image_tensor = torch.from_numpy(final_image.copy()).permute(2, 0, 1).float() / 255.0
        final_image_tensor = (final_image_tensor - IMG_MEAN) / IMG_STD

        max_val_in = np.max(final_input_psf)
        if max_val_in > 1e-7: final_input_psf = final_input_psf / max_val_in
        final_input_psf_tensor = torch.from_numpy(final_input_psf).float().unsqueeze(0)

        if not is_negative_sample: 
            target_psf_sum = np.sum(final_target_psf)
            if target_psf_sum > 1e-7:
                final_target_psf = final_target_psf / target_psf_sum
        final_target_psf_tensor = torch.from_numpy(final_target_psf).float().unsqueeze(0)
        
        confidence_target_tensor = torch.tensor(confidence_target, dtype=torch.float32)

        expected_shape = (model_input_size, model_input_size)
        if final_image_tensor.shape != (3, *expected_shape) or \
           final_input_psf_tensor.shape != (1, *expected_shape) or \
           final_target_psf_tensor.shape != (1, *expected_shape):
            print(f"Warning: Final shape mismatch for {image_path}. Retrying."); continue

        return final_image_tensor, final_input_psf_tensor, final_target_psf_tensor, confidence_target_tensor

    print(f"Warning: Failed to generate a valid sample after {max_retries} retries. Returning None tuple.")
    return None, None, None, None


def generate_batch(image_paths, gt_paths, batch_size, generation_fn=generate_train_sample, **kwargs):
    """Generates a batch of data including confidence targets."""
    image_batch, input_psf_batch, output_psf_batch, confidence_target_batch = [], [], [], []
    attempts = 0
    max_attempts = batch_size * 10 

    while len(image_batch) < batch_size and attempts < max_attempts:
        attempts += 1
        try:
            sample = generation_fn(image_paths, gt_paths, **kwargs) 

            if sample is not None and sample[0] is not None: 
                img, in_psf, out_psf, conf_tgt = sample
                if isinstance(img, torch.Tensor) and isinstance(in_psf, torch.Tensor) and \
                   isinstance(out_psf, torch.Tensor) and isinstance(conf_tgt, torch.Tensor):
                    image_batch.append(img)
                    input_psf_batch.append(in_psf)
                    output_psf_batch.append(out_psf)
                    confidence_target_batch.append(conf_tgt)
                else:
                    print(f"Warning: generation_fn returned non-Tensor data. Skipping. Types: {type(img)}, {type(in_psf)}, {type(out_psf)}, {type(conf_tgt)}")
        except Exception as e:
            import traceback
            print(f"Error during sample generation: {e}. Skipping sample.")
            print(traceback.format_exc())
            continue

    if not image_batch: 
        print(f"Warning: Failed to generate any valid samples for a batch after {max_attempts} attempts.")
        return None, None, None, None

    try:
        final_image_batch = torch.stack(image_batch)
        final_input_psf_batch = torch.stack(input_psf_batch)
        final_output_psf_batch = torch.stack(output_psf_batch)
        final_confidence_target_batch = torch.stack(confidence_target_batch) 
        return final_image_batch, final_input_psf_batch, final_output_psf_batch, final_confidence_target_batch
    except Exception as e:
        print(f"Error during torch.stack: {e}")
        if image_batch: print("Individual image shapes:", [t.shape for t in image_batch])
        if input_psf_batch: print("Individual input_psf shapes:", [t.shape for t in input_psf_batch])
        if output_psf_batch: print("Individual output_psf shapes:", [t.shape for t in output_psf_batch])
        if confidence_target_batch: print("Individual confidence target shapes:", [t.shape for t in confidence_target_batch])
        return None, None, None, None

if __name__ == "__main__":
    print("Running dataset.py test...")
    test_image_dir = IMAGE_DIR_TRAIN_VAL
    test_gt_dir = GT_DIR_TRAIN_VAL
    num_samples_to_show = 2 # Reduced for brevity, as skew is hard to see visually in single samples

    image_paths = sorted(glob.glob(os.path.join(test_image_dir, '*.jpg')))
    gt_paths = sorted(glob.glob(os.path.join(test_gt_dir, '*.mat'))) # Ensure these are aligned or logic in generate_train_sample handles mismatches
    if not image_paths: print(f"Error: No images found in {test_image_dir}"); exit()
    print(f"Found {len(image_paths)} images.")
    if not gt_paths: print(f"Warning: No GT .mat files found in {test_gt_dir}. Ensure paths are correct or image names match GTs.")


    # Test with different curriculum factors and skew powers
    curriculum_test_factors = [1.0, 0.5]
    skew_test_powers = [0.0, 2.0] # 0.0 for uniform, 2.0 for skewed

    for factor_idx, test_factor in enumerate(curriculum_test_factors):
        for skew_idx, test_skew_power in enumerate(skew_test_powers):
            print(f"\n--- Generating Samples with Curriculum Factor: {test_factor}, Skew Power: {test_skew_power} ---")
            for i in range(num_samples_to_show):
                print(f"\n--- Generating Sample {i+1} (Factor {test_factor}, Skew {test_skew_power}, negative_prob=0.3 for testing) ---")
                sample_data = generate_train_sample(
                    image_paths, gt_paths,
                    augment_size=AUGMENTATION_SIZE, model_input_size=MODEL_INPUT_SIZE,
                    psf_sigma=GT_PSF_SIGMA, negative_prob=0.3, # Test with higher neg prob
                    current_curriculum_crop_region_factor=test_factor,
                    target_point_skew_power=test_skew_power # Test with skew
                )

                if sample_data is None or sample_data[0] is None:
                    print("Failed to generate a valid sample. Skipping visualization.")
                    continue

                img_tensor, input_psf_tensor, target_psf_tensor, confidence_target_tensor = sample_data

                print(f"Generated Tensor Shapes:")
                print(f"  Image:      {img_tensor.shape}")
                print(f"  Input PSF:  {input_psf_tensor.shape}")
                print(f"  Target PSF: {target_psf_tensor.shape}")
                print(f"  Confidence Target: {confidence_target_tensor.shape}, Value: {confidence_target_tensor.item():.1f}")

                img_vis = img_tensor.cpu() * IMG_STD.cpu() + IMG_MEAN.cpu()
                img_vis = torch.clamp(img_vis, 0, 1)
                img_vis_np = img_vis.permute(1, 2, 0).numpy()        

                input_psf_np = input_psf_tensor.squeeze().cpu().numpy()
                target_psf_np = target_psf_tensor.squeeze().cpu().numpy()

                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                title_suffix = "POSITIVE" if confidence_target_tensor.item() == 1.0 else "NEGATIVE"
                fig.suptitle(f'Sample (Factor: {test_factor}, Skew: {test_skew_power}, {title_suffix})', fontsize=14)

                axes[0].imshow(img_vis_np)
                axes[0].set_title(f'Input Image ({MODEL_INPUT_SIZE}x{MODEL_INPUT_SIZE})')
                axes[0].axis('off')

                im_in = axes[1].imshow(input_psf_np, cmap='viridis', vmin=0)
                axes[1].set_title(f'Input PSF (Max: {np.max(input_psf_np):.4f})')
                axes[1].axis('off')
                fig.colorbar(im_in, ax=axes[1], fraction=0.046, pad=0.04)

                im_tgt = axes[2].imshow(target_psf_np, cmap='viridis', vmin=0)
                axes[2].set_title(f'Target PSF (Sum: {np.sum(target_psf_np):.4f})')
                axes[2].axis('off')
                fig.colorbar(im_tgt, ax=axes[2], fraction=0.046, pad=0.04)
                
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                save_path = f"sample_curriculum_factor_{test_factor}_skew_{test_skew_power}_sample_{i}.png"
                plt.savefig(save_path)
                print(f"Saved sample plot to {save_path}")
                plt.close()
    print("\nDataset test finished.")