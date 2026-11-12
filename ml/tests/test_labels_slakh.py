"""Slakh pitch labels: MIDI 440 Hz -> frames and FE-v1 band; speed warp of f0 (D-034)."""
import numpy as np
import pretty_midi
import torch

from avis_ml.data_index import save_midi_notes
from avis_ml.frontend import BANDS, band_centers
from avis_ml.mixer_v2 import RealMixer
from avis_ml.real_data import FRAMES, SLOTS, _f0_from_midi


def test_midi_440_hits_frames_and_band(tmp_path):
    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    inst = pretty_midi.Instrument(program=0, is_drum=False)
    inst.notes.append(pretty_midi.Note(velocity=100, pitch=69, start=0.5, end=1.5))
    pm.instruments.append(inst)
    midi = tmp_path / "a.mid"
    pm.write(str(midi))
    lab = tmp_path / "a.npz"
    assert save_midi_notes(midi, lab) == 1

    f0 = _f0_from_midi(str(lab), 0.0)
    assert f0.shape == (FRAMES, SLOTS)
    active = np.nonzero(f0[:, 0] > 0)[0]
    # 0.5..1.5 s on the 10 ms grid -> frames 50..149
    assert active[0] == 50 and active[-1] == 149 and len(active) == 100
    assert np.allclose(f0[active, 0], 440.0, atol=1e-3)
    assert f0[0, 0] == 0 and f0[200, 0] == 0

    # the FE-v1 band nearest to 440 Hz must be within +-20 cents (192 log bands)
    fc = band_centers()
    band = int(np.argmin(np.abs(np.log(fc) - np.log(440.0))))
    assert 0 <= band < BANDS
    assert abs(1200.0 * np.log2(fc[band] / 440.0)) < 20.0
    gauss = np.exp(-0.5 * ((np.log(fc) - np.log(440.0)) / 0.03) ** 2)
    assert int(np.argmax(gauss)) == band and gauss[band] > 0.9


def test_speed_warp_moves_f0_in_time():
    f0 = torch.zeros(FRAMES, SLOTS)
    f0[100, 0] = 440.0
    # audio is read at t*speed; a note at frame 100 sounds at 100/speed
    for speed, expect in ((1.1, 91), (0.9, 111)):
        out = RealMixer._warp_f0(f0, speed) * speed
        assert int(torch.argmax(out[:, 0])) == expect
        assert abs(float(out[expect, 0]) - 440.0 * speed) < 1e-3
    same = RealMixer._warp_f0(f0, 1.0)
    assert torch.equal(same, f0)
