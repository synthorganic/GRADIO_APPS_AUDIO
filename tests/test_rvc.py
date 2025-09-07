import pytest
import musicgen_stems_continue2 as msc2


def test_rvc_raises_when_unavailable(tmp_path):
    assert not msc2.RVC_AVAILABLE
    vocal = tmp_path / "vocal.wav"
    model = tmp_path / "model.pth"
    out = tmp_path / "out.wav"
    with pytest.raises(RuntimeError):
        msc2.rvc_convert_vocals(vocal, model, out)
