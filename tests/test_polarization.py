import pytest
import imaging.polarization as mod

CASES = [
    ("polar_angle_perpixel_align", (1,)),
    ("polar_angle_base_unify", (1,)),
    ("material_adaptive_thresh", (1,)),
    ("frame_transmittance_calib", (1,)),
    ("polar_overlap_joint_solve", (1,)),
    ("polar_buffer_clear", (1,)),
    ("load_polar_stack", (1,)),
    ("align_polar_frames", (1,)),
    ("separate_reflection", (1,)),
    ("present_polar", (1,)),
    ("validate_polar", (1,)),
    ("metrics_polar", (1,)),
    ("audit_polar", (1,)),
    ("fallback_polar", (1,)),
    ("determinism_polar", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
