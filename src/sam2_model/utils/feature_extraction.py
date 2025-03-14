import autorootcwd
import torch
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from einops import rearrange

def preprocess_image(image_path, device):
    """
    Preprocess the image for the image encoder.
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (1024, 1024), interpolation=cv2.INTER_LINEAR)
    image = image.astype(np.float32) / 255.0
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    return {"pixel_values": image.to(device)}

def compute_pca_feature_image_encoder(feature_map, original_size):
    """
    Compute the PCA feature of the image encoder.
    Args:
        feature_map: (B, H, W, C)
        original_size: (H, W)
    Returns:
        E_pca_3_resized: (H, W, 3)
    """
    B, H, W, C = feature_map.shape  # (B, H, W, C)
    reshaped_feature = feature_map.reshape(B, H * W, C) # (B, H*W, C)
    E_patch_norm = reshaped_feature.squeeze(0)  # (H*W, C)

    # PCA
    _, _, V = torch.pca_lowrank(E_patch_norm)
    E_pca_3 = torch.matmul(E_patch_norm, V[:, :3])  # (H*W, 3)
    E_pca_3 = (E_pca_3 - E_pca_3.min()) / (E_pca_3.max() - E_pca_3.min())  # Min-Max normalization

    # reshape to original image size
    feature_size = int(np.sqrt(E_pca_3.shape[0]))
    E_pca_3 = rearrange(E_pca_3, "(h w) c -> h w c", h=feature_size, w=feature_size)
    E_pca_3_resized = cv2.resize(E_pca_3.cpu().numpy(), (original_size[1], original_size[0]), interpolation=cv2.INTER_CUBIC)    # h, w, c -> w, h, c
    return E_pca_3_resized

def compute_pca_feature_mask_decoder(mode, feature_map, original_size):
    """
    Compute the PCA feature of the mask decoder.
    Args:
        mode: "image" or "stream"
        feature_map: (B, H*W, C) or (H*W, C)
        original_size: (H, W)
    Returns:
        E_pca_3_resized: (H, W, 3)
    """
    if feature_map.dim() == 2:
        E_patch = feature_map.unsqueeze(0)  # (1, 4096, 256)
    else:
        E_patch = feature_map  # (B, H*W, C)
    
    E_patch = E_patch.squeeze(0)  # (4096, 256)

    if mode == "image":
        _, _, V = torch.pca_lowrank(E_patch)
    
    elif mode == "stream":
        _, _, V = torch.pca_lowrank(E_patch.cpu())
        V = V.cpu()
        E_patch = E_patch.cpu()
    
    E_pca_3 = torch.matmul(E_patch, V[:, :3]) # (4096, 256) x (256, 3) -> (4096, 3)
    E_pca_3 = (E_pca_3 - E_pca_3.min()) / (E_pca_3.max() - E_pca_3.min() + 1e-8)
    
    # reshape to original image size (4096=64x64)
    feature_size = int(np.sqrt(E_pca_3.shape[0]))
    if feature_size * feature_size != E_pca_3.shape[0]:
        print(feature_size, feature_size * feature_size, E_pca_3.shape[0])
        raise ValueError("feature size is not square")

    E_pca_3 = rearrange(E_pca_3, "(h w) c -> h w c", h=feature_size, w=feature_size) # (4096, 3) -> (64, 64, 3)
    E_pca_3_resized = cv2.resize(E_pca_3.cpu().numpy(), (original_size[1], original_size[0]), interpolation=cv2.INTER_CUBIC)

    return E_pca_3_resized

def apply_pca_and_visualize(mode, feature_map, original_image, feature_type, save_path = None, visualize = False):
    """
    Apply PCA and visualize the feature.
    Args:
        mode: "image" or "stream"
        feature_map: (B, H, W, C) or (H*W, C)
        original_image: (H, W, 3)
        feature_type: "mask_decoder" or "image_encoder"
        save_path: save path
    Returns:
        E_pca_3_resized: (H, W, 3)
    """

    if feature_type == "mask_decoder":
        E_pca_3_resized = compute_pca_feature_mask_decoder(mode, feature_map, original_size=original_image.shape[:2])
    elif feature_type == "image_encoder":
        E_pca_3_resized = compute_pca_feature_image_encoder(mode, feature_map, original_size=original_image.shape[:2])
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(original_image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    
    axes[1].imshow(np.clip(E_pca_3_resized, 0, 1))
    axes[1].set_title(feature_type)
    axes[1].axis("off")
    
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    if visualize:
        plt.show()
    plt.close()
    
    return E_pca_3_resized

def plot_combined_features(original_image, feature_list, save_path):
    """
    Plot the combined features.
    Args:
        original_image: (H, W, 3)
        feature_list: dictionary of features
        save_path: save path
    """
    total_images = 1 + len(feature_list)
    ncols = 3
    nrows = math.ceil(total_images / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = axes.flatten()

    axes[0].imshow(original_image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    idx = 1
    for name, pca_img in feature_list.items():
        axes[idx].imshow(pca_img)
        axes[idx].set_title(name)
        axes[idx].axis("off")
        idx += 1

    for ax in axes[idx:]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()
