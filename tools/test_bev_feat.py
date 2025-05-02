import torch
import math
import torch.nn.functional as F
from torchvision.transforms.functional import affine
from torchvision.transforms.functional import InterpolationMode
import matplotlib.pyplot as plt
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Align and compare BEV features')
    parser.add_argument('--result0', type=str, default='workspace/bev_similarity/bev_embed_scene-0014_2_2.pth', help='Path to first BEV feature file')
    parser.add_argument('--result1', type=str, default='workspace/bev_similarity/bev_embed_scene-0014_3_3image.png.pth', help='Path to second BEV feature file')
    parser.add_argument('--output', type=str, default='workspace/bev_similarity/bev_similarity_heatmap_scene-0014.png', help='Path to save similarity heatmap')
    return parser.parse_args()

def main():
    args = parse_args()

    result_0 = torch.load(args.result0)
    result_1 = torch.load(args.result1)

    pos_0 = result_0['pos']
    angle_0 = result_0['angle']
    bev_embed_0 = result_0['bev_embed']

    pos_1 = result_1['pos']
    angle_1 = result_1['angle']
    bev_embed_1 = result_1['bev_embed']

    # Calculate displacement of result_1 relative to result_0
    dx_world = pos_1[0] - pos_0[0]
    dy_world = pos_1[1] - pos_0[1]

    # Convert angles to radians
    angle_0_rad = angle_0 * math.pi / 180.0
    angle_1_rad = angle_1 * math.pi / 180.0

    # Calculate relative rotation between two ego coordinate systems
    relative_angle = angle_1_rad - angle_0_rad

    # Transform displacement to result_0's ego coordinate system
    dx_ego = dx_world * math.cos(angle_0_rad) + dy_world * math.sin(angle_0_rad)
    dy_ego = -dx_world * math.sin(angle_0_rad) + dy_world * math.cos(angle_0_rad)

    # Reshape BEV features
    bev_embed_0 = bev_embed_0.reshape(200, 200, -1).permute(2, 0, 1)
    bev_embed_1 = bev_embed_1.reshape(200, 200, -1).permute(2, 0, 1)

    # Calculate pixel displacement (assuming each pixel represents 0.5 meters)
    pixel_dx = dy_ego / 0.5  # Note: swap dx and dy due to different coordinate systems between image and ego
    pixel_dy = dx_ego / 0.5

    # Align BEV features
    aligned_bev_embed = affine(
        bev_embed_1,  # Align result_1's features to result_0's space
        angle=-relative_angle * 180 / math.pi,  # Convert relative rotation to degrees and apply negative rotation
        translate=[-pixel_dx, -pixel_dy],  # Translation amount
        scale=1.0,
        shear=0.0,
        interpolation=InterpolationMode.BILINEAR,
    )

    # Transform back to original shape
    aligned_bev_embed = aligned_bev_embed.permute(1, 2, 0)
    bev_embed_0 = bev_embed_0.permute(1, 2, 0)

    # Calculate feature similarity
    bev0_normalized = F.normalize(bev_embed_0, p=2, dim=-1)
    bev1_normalized = F.normalize(aligned_bev_embed, p=2, dim=-1)
    similarity_map = torch.sum(bev0_normalized * bev1_normalized, dim=-1)

    # Visualize similarity map
    similarity_np = similarity_map.cpu().numpy()
    plt.figure(figsize=(10, 10))
    plt.imshow(similarity_np, cmap='hot', vmin=0, vmax=1)
    plt.colorbar(label='Cosine Similarity')
    plt.title('Pixel-wise Feature Similarity Heatmap')
    plt.scatter(100, 100, c='green', s=50, marker='o', label='Ego')
    plt.legend()
    plt.savefig(args.output, dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    main()
