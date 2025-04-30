import torch
import math
import torch.nn.functional as F
from torchvision.transforms.functional import affine
from torchvision.transforms.functional import InterpolationMode
import matplotlib.pyplot as plt

result_0 = torch.load('bev_embed_0_9.pth')
result_1 = torch.load('bev_embed_6_15.pth')

pos_0 = result_0['pos']
angle_0 = result_0['angle']
bev_embed_0 = result_0['bev_embed']

pos_1 = result_1['pos']
angle_1 = result_1['angle']
bev_embed_1 = result_1['bev_embed']

dx_world = pos_1[0] - pos_0[0]
dy_world = pos_1[1] - pos_0[1]

rotation_angle = angle_0
rotation_angle = rotation_angle * math.pi / 180.0

dx_ego = dx_world * math.cos(rotation_angle) + dy_world * math.sin(rotation_angle)
dy_ego = -dx_world * math.sin(rotation_angle) + dy_world * math.cos(rotation_angle)

bev_embed_0 = bev_embed_0.reshape(200, 200, -1).permute(2, 0, 1)
bev_embed_1 = bev_embed_1.reshape(200, 200, -1)

pixel_dy = dx_ego / 0.5
pixel_dx = dy_ego / 0.5

aligned_bev_embed = affine(
    bev_embed_0,
    angle=0.0,  # No rotation
    translate=[-pixel_dx, -pixel_dy],  # Negative because we align 0_9 to 6_15
    scale=1.0,
    shear=0.0,
    interpolation=InterpolationMode.BILINEAR,
)

aligned_bev_embed = aligned_bev_embed.permute(1, 2, 0)
bev1_normalized = F.normalize(aligned_bev_embed, p=2, dim=-1)  # L2归一化
bev2_normalized = F.normalize(bev_embed_1, p=2, dim=-1)

similarity_map = torch.sum(bev1_normalized * bev2_normalized, dim=-1)

similarity_np = similarity_map.cpu().numpy()

# 创建热力图
plt.figure(figsize=(10, 10))
plt.imshow(similarity_np, cmap='hot', vmin=0, vmax=1)  # 'hot'/'viridis'/'plasma'
plt.colorbar(label='Cosine Similarity')
plt.title('Pixel-wise Feature Similarity Heatmap')

# 标记自车位置（可选）
plt.scatter(100, 100, c='green', s=50, marker='o', label='Ego')  # 假设ego在(100,100)
plt.legend()

# 保存或显示
plt.savefig('bev_similarity_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()
