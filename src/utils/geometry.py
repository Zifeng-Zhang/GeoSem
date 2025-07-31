import torch


def inverse_extrinsics(Tcw: torch.Tensor) -> torch.Tensor:
    """
    Invert camera-from-world extrinsics to obtain world-from-camera.
    Tcw: (..., 4, 4) OpenCV convention (camera = R * world + t).
    Returns Twc: (..., 4, 4).
    """
    R = Tcw[..., :3, :3]
    t = Tcw[..., :3, 3:4]
    Rt = R.transpose(-1, -2)
    tw = -Rt @ t
    Twc = torch.eye(4, device=Tcw.device, dtype=Tcw.dtype).expand_as(Tcw).clone()
    Twc[..., :3, :3] = Rt
    Twc[..., :3, 3] = tw[..., 0]
    return Twc


def farthest_point_sampling(xyz: torch.Tensor, m: int, seed: int = 0) -> torch.Tensor:
    """
    Naive FPS in O(N*M) for PoC (N up to ~2x quota). Returns indices of selected points.
    xyz: (N, 3) in any metric space you care about (camera/world coords).
    m: number of points to keep
    """
    N = xyz.shape[0]
    m = min(m, N)
    if m <= 0:
        return torch.empty(0, dtype=torch.long, device=xyz.device)

    # initialize
    torch.manual_seed(seed)
    idxs = torch.empty(m, dtype=torch.long, device=xyz.device)
    dists = torch.full((N,), 1e10, dtype=xyz.dtype, device=xyz.device)

    # start from a random point
    farthest = torch.randint(0, N, (1,), device=xyz.device).item()
    for i in range(m):
        idxs[i] = farthest
        # update distances
        dist = torch.sum((xyz - xyz[farthest]) ** 2, dim=1)
        dists = torch.minimum(dists, dist)
        farthest = torch.argmax(dists).item()
    return idxs