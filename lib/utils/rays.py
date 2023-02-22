import numpy as np

def get_ray_directions(W, H, fx, fy, cx, cy, use_pixel_centers=True, mode='OPENCV'):
    pixel_center = 0.5 if use_pixel_centers else 0
    if mode == 'OPENCV':
        j, i = np.meshgrid(
            np.arange(H, dtype=np.float32) + pixel_center,
            np.arange(W, dtype=np.float32) + pixel_center,
            indexing='ij'
        )
        directions = np.stack([(i - cx) / fx, (j - cy) / fy, np.ones_like(i)], -1) # (H, W, 3)
    elif mode == 'OPENGL':
        i, j = np.meshgrid(
            np.arange(W, dtype=np.float32) + pixel_center,
            np.arange(H, dtype=np.float32) + pixel_center,
            indexing='xy'
        )
        directions = np.stack([(i - cx) / fx, -(j - cy) / fy, -np.ones_like(i)], -1) # (H, W, 3)
    return directions

def get_rays(directions, c2w, keepdim=False):
    # Rotate ray directions from camera coordinate to the world coordinate
    # rays_d = directions @ c2w[:, :3].T # (H, W, 3) # slow?
    assert directions.shape[-1] == 3

    if directions.ndim == 2: # (N_rays, 3)
        assert c2w.ndim == 3 # (N_rays, 4, 4) / (1, 4, 4)
        rays_d = (directions[:,None,:] * c2w[:,:3,:3]).sum(-1) # (N_rays, 3)
        if c2w.shape[0] == 1:
            rays_o = np.repeat(c2w[:,:,3], rays_d.shape[0], 0)
        else:
            rays_o = c2w[:,:,3]
    elif directions.ndim == 3: # (H, W, 3)
        if c2w.ndim == 2: # (4, 4)
            rays_d = (directions[:,:,None,:] * c2w[None,None,:3,:3]).sum(-1) # (H, W, 3)
            rays_o = np.repeat(c2w[None,None,:,3],rays_d.shape[0], axis=0)
            rays_o = np.repeat(rays_o,rays_d.shape[1], axis=1)
        elif c2w.ndim == 3: # (B, 4, 4)
            rays_d = (directions[None,:,:,None,:] * c2w[:,None,None,:3,:3]).sum(-1) # (B, H, W, 3)
            rays_o = np.repeat(c2w[:,None,None,3],rays_d.shape[1], axis=1)
            rays_o = np.repeat(rays_o,rays_d.shape[2], axis=2)

    if not keepdim:
        rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)

    return rays_o, rays_d