"""Tests for the dataset category mapping and the 1-second activity helper."""
import numpy as np

from avis_ml.categories import MUSDB_STREAMS, VOCAB, musdb_category, slakh_category
from avis_ml.data_index import active_seconds

SR = 44100


def test_vocab_is_fixed_and_unique():
    assert len(VOCAB) == 20 and len(set(VOCAB)) == 20
    assert VOCAB[0] == "vocal_lead" and VOCAB[-1] == "fx"


def test_musdb_only_vocals_and_bass_become_labels():
    assert musdb_category("vocals") == "vocal_lead"
    assert musdb_category("bass") == "bass"
    assert musdb_category("drums") is None
    assert musdb_category("other") is None
    assert MUSDB_STREAMS["vocals"] == 4 and MUSDB_STREAMS["mix"] == 0


def test_slakh_mapping_by_program():
    assert slakh_category("Guitar", 30) == "guitar"
    assert slakh_category("Bass", 33) == "bass"
    assert slakh_category("Piano", 1) == "piano"
    assert slakh_category("Piano", 6) == "keys"
    assert slakh_category("Synth Lead", 81) == "synth_lead"
    assert slakh_category("Strings (continued)", 52) == "strings"
    assert slakh_category("Reed", 66) == "winds"
    assert slakh_category("Synth Pad", 90) == "synth_pad"
    assert slakh_category("Unknown Thing", 5) == "other_tonal"


def test_active_seconds_threshold_and_alignment():
    quiet = np.zeros(SR * 3, dtype=np.float32)
    loud = (0.1 * np.sin(2 * np.pi * 440 * np.arange(SR * 2) / SR)).astype(np.float32)
    x = np.concatenate([quiet, loud])  # 3 s silence, then 2 s tone at about -23 dBFS
    assert active_seconds(x, SR) == [3, 4]
    stereo = np.stack([x, x], axis=1)
    assert active_seconds(stereo, SR) == [3, 4]
    assert active_seconds(np.zeros(SR // 2, dtype=np.float32), SR) == []
