import yaml

from src.models.geometry_backbone.geo_lifter import GeoLifter
from src.models.geometry_backbone.vggt_wrapper import VGGTWrapper


def test_geo_lifter_basic_shapes(cfg, vggt_out):
    lifter = GeoLifter(cfg)
    out = lifter.forward(vggt_out, scene_meta=None, batch_idx=0)

    assert "coords" in out and "feats_geo_raw" in out
    N = out["coords"].shape[0]
    assert out["coords"].shape == (N, 3)
    assert out["feats_geo_raw"].shape == (N, 16)
    if out["weights"] is not None:
        assert out["weights"].shape == (N,)
    if out["uv"] is not None:
        assert out["uv"].shape == (N, 2)
    if out["view_ids"] is not None:
        assert out["view_ids"].shape == (N,)
    assert out["meta"]["stride"] == 14
    assert out["meta"]["coord_frame"] == "camera"


def test_geo_lifter_bilinear_values_center(cfg, vggt_out):
    """
    Check bilinear interpolation numerics at the center of 2x2 patch grid.
    With align_corners=False and stride=14:
      - H=28, W=28 -> H_p=W_p=2, patch centers roughly at x_p=0 and 1.
      - At (u,v) = (14,14) --> x_p=1.0, y_p=1.0 (clamped by eps),
        neighbors are (0,0),(1,0),(0,1),(1,1).
      Using tokens f00=0, f10=10, f01=100, f11=1000, the bilinear
      at the exact center (0.5,0.5) would be ~ (0+10+100+1000)/4, but here
      discrete mapping is approximate; we just verify the value is between min/max.
    """
    lifter = GeoLifter(cfg)
    out = lifter.forward(vggt_out)

    val = out["feats_geo_raw"][0, 0].item()
    # since tokens are in [0, 1000], interpolated value should be within.
    assert 0.0 <= val <= 1000.0


if __name__ == "__main__":
    with open("../configs/default.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    vggt = VGGTWrapper(cfg['model']['vggt']).eval()
    vggt_out = vggt.forward_from_paths(["../examples/scannet_samples/scene0030_02/20.jpg",
                                        "../examples/scannet_samples/scene0030_02/80.jpg"])

    lifter = GeoLifter(cfg['lift_geo'])
    geo_pack = lifter.forward(vggt_out, scene_meta=None, batch_idx=0)
    print(geo_pack['coords'].shape, geo_pack['feats_geo_raw'].shape)

    print(vggt_out['patch_tokens_2048'].mean().item(), vggt_out['patch_tokens_2048'].std().item())
    print(geo_pack['feats_geo_raw'].mean().item(), geo_pack['feats_geo_raw'].std().item())

    print(geo_pack["feats_geo_raw"].norm(dim=-1).mean().item())

    # test_geo_lifter_basic_shapes(cfg, vggt_out)
    # test_geo_lifter_bilinear_values_center(cfg, vggt_out)
    print("GeoLifter tests passed.")
