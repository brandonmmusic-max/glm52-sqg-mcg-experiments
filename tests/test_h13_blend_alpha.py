from scripts.encode_expert_local_h13_shard import _fixed_alpha_construction


def test_fixed_alpha_construction_is_exact_and_stable() -> None:
    assert _fixed_alpha_construction(None).endswith("cap_0p75_v1")
    assert _fixed_alpha_construction(0.0).endswith(
        "fixed_alpha_0_layer_global_prior_v2"
    )
    assert _fixed_alpha_construction(0.25).endswith(
        "fixed_alpha_0p25_layer_global_prior_v2"
    )
    assert _fixed_alpha_construction(0.5).endswith(
        "fixed_alpha_0p5_layer_global_prior_v2"
    )
    assert _fixed_alpha_construction(0.75).endswith(
        "fixed_alpha_0p75_layer_global_prior_v2"
    )
    assert _fixed_alpha_construction(1.0).endswith(
        "fixed_alpha_1_layer_global_prior_v2"
    )
