# iterate_on_single_patch_with_iterative_inference.py
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from scipy.ndimage import gaussian_filter

# --- Import from your project's config and model files ---
try:
    from config import (
        DEVICE, MODEL_INPUT_SIZE, BEST_MODEL_PATH, GT_PSF_SIGMA
        # Add IMG_MEAN_CPU, IMG_STD_CPU here if they are in your config.py
    )
    print("Successfully imported configuration from config.py")
except ImportError:
    print("ERROR: Could not import from config.py. Please ensure it exists in the same directory and defines:")
    print("  DEVICE, MODEL_INPUT_SIZE, BEST_MODEL_PATH, GT_PSF_SIGMA")
    print("Using fallback default values (may not be correct for your model).")
    # Fallback defaults (adjust if necessary, but ideally fix config.py)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_INPUT_SIZE = 224
    BEST_MODEL_PATH = "path/to/your/best_model.pth" # THIS WILL LIKELY FAIL - FIX config.py
    GT_PSF_SIGMA = 4.0
    # exit() # Optionally exit if config is crucial

try:
    from model import VGG19FPNASPP
    print("Successfully imported VGG19FPNASPP from model.py")
except ImportError:
    print("ERROR: Could not import VGG19FPNASPP from model.py.")
    print("Please ensure model.py is in the same directory and defines the VGG19FPNASPP class.")
    exit() # Critical, cannot proceed without the model definition

# ImageNet Mean/Std for Normalization/Unnormalization
IMG_MEAN_CPU = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMG_STD_CPU = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# --- Helper Functions (from test_full_image_patched_inference.py) ---
def create_input_psf_from_points(points_list, shape, sigma):
    h, w = shape
    delta_map = np.zeros((h, w), dtype=np.float32)
    if not points_list:
        return delta_map # Returns a zero map if no points
    for x, y in points_list:
        x_coord = np.clip(int(round(x)), 0, w - 1)
        y_coord = np.clip(int(round(y)), 0, h - 1)
        delta_map[y_coord, x_coord] += 1.0
    input_psf = gaussian_filter(delta_map, sigma=sigma, order=0, mode='constant', cval=0.0)
    max_val = np.max(input_psf)
    if max_val > 1e-7: # Normalize to 0-1 range
        input_psf /= max_val
    return input_psf

# --- Main Script Logic ---

# Path to the image
image_file_path = r"C:\Users\Mehmet_Postdoc\Desktop\datasets_for_experiments\ShanghaiTech_Crowd_Counting_Dataset\part_A_final\test_data\images\IMG_115.jpg" #Example Path

# Load the image
image = cv2.imread(image_file_path)
if image is None:
    print(f"Error: Could not load image from {image_file_path}")
    exit()

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
print("Original image shape:", image_rgb.shape)

initial_patch_height = 224
initial_patch_width = 224
img_height, img_width, _ = image_rgb.shape
patches_orig_size = []

for r in range(0, img_height - initial_patch_height + 1, initial_patch_height):
    for c in range(0, img_width - initial_patch_width + 1, initial_patch_width):
        patch = image_rgb[r:r + initial_patch_height, c:c + initial_patch_width]
        patches_orig_size.append(patch)

print(f"Number of {initial_patch_width}x{initial_patch_height} patches extracted: {len(patches_orig_size)}")

if not patches_orig_size:
    print("No patches were extracted. Exiting.")
    exit()

patches_resized = [cv2.resize(patch, (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)) for patch in patches_orig_size]
print(f"Resized patches to {MODEL_INPUT_SIZE}x{MODEL_INPUT_SIZE}.")

patch_index_to_test = 0
if patch_index_to_test >= len(patches_resized):
    print(f"Error: Patch index {patch_index_to_test} is out of bounds. Max index is {len(patches_resized)-1}.")
    print(f"Using patch index 0 instead.")
    patch_index_to_test = 0
    if not patches_resized:
        print("No patches available. Exiting.")
        exit()

selected_patch_np = patches_resized[patch_index_to_test] # This is uint8, 0-255

if not os.path.exists(BEST_MODEL_PATH):
    print(f"CRITICAL ERROR: Model weights not found at '{BEST_MODEL_PATH}' (from config.py).")
    print("Please ensure BEST_MODEL_PATH in config.py is correct.")

    if BEST_MODEL_PATH == "path/to/your/best_model.pth": # Specific check for default placeholder
        print("Using a placeholder model path. Inference will likely fail if not updated.")
        print("Attempting to proceed without loading weights (model will be initialized randomly).")
        model_loaded_successfully = False
    else:
        exit()
else:
    model_loaded_successfully = True


model = VGG19FPNASPP().to(DEVICE)

if model_loaded_successfully:
    try:
        model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
        print(f"Model loaded successfully from {BEST_MODEL_PATH} to {DEVICE}.")
    except Exception as e:
        print(f"Error loading model state_dict from {BEST_MODEL_PATH}: {e}")
        print("Ensure the model definition in model.py matches the weights file.")
        print("Proceeding with an uninitialized model for structure testing if possible.")
        model_loaded_successfully = False # Crucial for actual inference
else:
     print("Proceeding with randomly initialized model (weights not loaded).")

model.eval()

patch_tensor_np_float = selected_patch_np.astype(np.float32) / 255.0 # Normalize to 0-1 for model
patch_tensor_chw = torch.from_numpy(patch_tensor_np_float).permute(2, 0, 1) # This is on CPU

patch_tensor_norm = (patch_tensor_chw - IMG_MEAN_CPU) / IMG_STD_CPU
image_patch_tensor_batch = patch_tensor_norm.unsqueeze(0).to(DEVICE)

# --- Iterative Inference Loop ---
print(f"\nStarting iterative inference on patch {patch_index_to_test}...")
predicted_points_on_patch = [] # Stores (x,y) for PSF generation in next iteration
all_predictions_with_scores = [] # Stores ((x,y), score) for all predictions
num_iterations = 5 # Number of points to attempt to predict. User can change this (e.g., to 20).

# Storage for iteration-wise visualization
iterative_visualization_data = []

for iter_num in range(num_iterations):
    current_input_psf_np = create_input_psf_from_points(
        predicted_points_on_patch, 
        shape=(MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
        sigma=GT_PSF_SIGMA
    )
    current_input_psf_tensor = torch.from_numpy(current_input_psf_np).float().unsqueeze(0).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        predicted_output_psf_tensor, predicted_confidence_logits = model(image_patch_tensor_batch, current_input_psf_tensor)

    confidence_score = torch.sigmoid(predicted_confidence_logits).item()
    output_psf_np = predicted_output_psf_tensor.squeeze().cpu().numpy()
    max_yx = np.unravel_index(np.argmax(output_psf_np), output_psf_np.shape)
    pred_y, pred_x = max_yx[0], max_yx[1]

    predicted_points_on_patch.append((pred_x, pred_y))
    all_predictions_with_scores.append(((pred_x, pred_y), confidence_score))

    # Store data for this iteration's visualization
    iterative_visualization_data.append({
        "input_psf": current_input_psf_np.copy(), 
        "output_psf": output_psf_np.copy(),       
        "predicted_point_xy": (pred_x, pred_y),
        "confidence": confidence_score,
        "cumulative_points_xy": list(predicted_points_on_patch) # Copy of current list of all points found so far
    })

    print(f"  Iteration {iter_num + 1}/{num_iterations}: New point ({pred_x}, {pred_y}), Confidence: {confidence_score:.4f}")


print("\nIterative inference finished.")
print(f"Total raw predictions made on patch {patch_index_to_test}: {len(all_predictions_with_scores)}")

# --- Visualization of each iteration's input/output ---
print("\nVisualizing each iteration's input PSF, output PSF, and cumulative points on patch...")
num_viz_iterations = len(iterative_visualization_data)
if num_viz_iterations > 0:
    max_rows_per_figure = 5 # Max number of iterations (rows) to show per figure
    num_figures = (num_viz_iterations + max_rows_per_figure - 1) // max_rows_per_figure

    for fig_idx in range(num_figures):
        start_iter_viz = fig_idx * max_rows_per_figure
        end_iter_viz = min((fig_idx + 1) * max_rows_per_figure, num_viz_iterations)
        current_num_rows_viz = end_iter_viz - start_iter_viz

        if current_num_rows_viz == 0: continue

        # Create figure and axes; squeeze=False ensures axes_iter is always 2D
        fig_iter, axes_iter = plt.subplots(current_num_rows_viz, 3, 
                                           figsize=(15, 4.5 * current_num_rows_viz), # width, height
                                           squeeze=False)
        
        fig_iter.suptitle(f"Iterative Inference: Patch {patch_index_to_test}, Iterations {start_iter_viz + 1}-{end_iter_viz} (of {num_viz_iterations})", 
                          fontsize=16)

        for i_row_viz in range(current_num_rows_viz):
            iter_idx_actual_data = start_iter_viz + i_row_viz # Index in iterative_visualization_data
            data = iterative_visualization_data[iter_idx_actual_data]
            iter_display_num = iter_idx_actual_data + 1 # 1-based iteration number for titles

            ax_in_psf   = axes_iter[i_row_viz, 0]
            ax_out_psf  = axes_iter[i_row_viz, 1]
            ax_patch_pts = axes_iter[i_row_viz, 2]

            # Plot 1: Input PSF
            ax_in_psf.imshow(data["input_psf"], cmap='viridis', vmin=0.0, vmax=1.0) # Normalized 0-1
            ax_in_psf.set_title(f"Iter {iter_display_num}: Input PSF to Model")
            ax_in_psf.axis('off')

            # Plot 2: Output PSF from Model
            # imshow auto-scales colors unless vmin/vmax are set. This is usually fine for visualizing peaks.
            im_out_psf = ax_out_psf.imshow(data["output_psf"], cmap='viridis') 
            ax_out_psf.scatter(data["predicted_point_xy"][0], data["predicted_point_xy"][1], s=60, c='red', marker='x')
            ax_out_psf.set_title(f"Iter {iter_display_num}: Output PSF (Conf: {data['confidence']:.2f})")
            ax_out_psf.axis('off')
            # Optional: Add a colorbar for the output PSF
            # fig_iter.colorbar(im_out_psf, ax=ax_out_psf, fraction=0.046, pad=0.04)


            # Plot 3: Selected Patch with Cumulative Points
            ax_patch_pts.imshow(selected_patch_np) # Display the original resized patch (uint8)
            
            cumulative_points_for_iter = np.array(data["cumulative_points_xy"])
            if cumulative_points_for_iter.size > 0:
                # Points from previous iterations for this patch
                if len(cumulative_points_for_iter) > 1:
                    ax_patch_pts.scatter(cumulative_points_for_iter[:-1, 0], cumulative_points_for_iter[:-1, 1], 
                                         s=30, c='yellow', marker='x', alpha=0.8, label='Previous Points')
                # Point predicted in the current iteration
                ax_patch_pts.scatter(cumulative_points_for_iter[-1, 0], cumulative_points_for_iter[-1, 1], 
                                     s=50, c='red', marker='o', label='Current Point')
                ax_patch_pts.legend(fontsize='small', loc='best')
            
            ax_patch_pts.set_title(f"Iter {iter_display_num}: Patch + Points ({len(cumulative_points_for_iter)} total)")
            ax_patch_pts.axis('off')

        fig_iter.tight_layout(rect=[0, 0.03, 1, 0.95]) # rect=[left, bottom, right, top]
        plt.show()
else:
    print("No iterations were run, or no data collected for visualization.")


# --- Filter points for final display based on confidence ---
CONFIDENCE_THRESHOLD_FOR_DISPLAY = 0.5  # Adjust this threshold as needed
points_to_display_on_patch = []
for point_coords, score in all_predictions_with_scores:
    if score >= CONFIDENCE_THRESHOLD_FOR_DISPLAY:
        points_to_display_on_patch.append(point_coords)

print(f"Number of points on patch {patch_index_to_test} with confidence >= {CONFIDENCE_THRESHOLD_FOR_DISPLAY:.2f}: {len(points_to_display_on_patch)}")


# --- Final Plotting ---
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(image_rgb)
plt.title("Original Full Image")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(selected_patch_np) # Show the resized patch (uint8) that was used for inference
patch_title = f"Selected Patch {patch_index_to_test} ({MODEL_INPUT_SIZE}x{MODEL_INPUT_SIZE})\n"
patch_title += f"Final Points (Conf >= {CONFIDENCE_THRESHOLD_FOR_DISPLAY:.2f})"

if points_to_display_on_patch:
    final_points_np = np.array(points_to_display_on_patch)
    plt.scatter(final_points_np[:, 0], final_points_np[:, 1], s=30, c='lime', marker='x', label=f'Conf >= {CONFIDENCE_THRESHOLD_FOR_DISPLAY:.2f}')
    patch_title += f": {len(points_to_display_on_patch)} shown"
    plt.legend(loc='best')
else:
    patch_title += f": None"
    plt.text(MODEL_INPUT_SIZE / 2, MODEL_INPUT_SIZE / 2,
             f"No points with\nconfidence >= {CONFIDENCE_THRESHOLD_FOR_DISPLAY:.2f}",
             horizontalalignment='center', verticalalignment='center',
             fontsize=9, color='red', bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))

plt.title(patch_title)
plt.axis('off')
plt.tight_layout()
plt.show()