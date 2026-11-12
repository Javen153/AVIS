"""Fixed category vocabulary (SPEC F) and dataset-to-category mapping.

Slakh: category from `inst_class` in metadata.yaml, refined by `program_num`.
Drum stems (`is_drum`) are never tonal labels and are not indexed.
MUSDB18-HQ: only `vocals` -> vocal_lead and `bass` -> bass; drums/other are mixes.
"""
from __future__ import annotations

VOCAB = [
    "vocal_lead", "vocal_back", "bass", "guitar", "keys", "piano", "synth_lead",
    "synth_pad", "strings", "brass", "winds", "organ", "kick", "snare", "hihat",
    "cymbal", "tom", "perc", "other_tonal", "fx",
]
PITCHED = {
    "vocal_lead", "vocal_back", "bass", "guitar", "keys", "piano", "synth_lead",
    "synth_pad", "strings", "brass", "winds", "organ", "other_tonal", "fx",
}

_CLASS_TO_CAT = {
    "Guitar": "guitar",
    "Piano": "piano",
    "Bass": "bass",
    "Strings (continued)": "strings",
    "Strings": "strings",
    "Brass": "brass",
    "Reed": "winds",
    "Pipe": "winds",
    "Synth Lead": "synth_lead",
    "Synth Pad": "synth_pad",
    "Organ": "organ",
    "Chromatic Percussion": "other_tonal",
    "Ethnic": "other_tonal",
    "Sound effects": "fx",
    "Sound Effects": "fx",
    "Percussive": "perc",
}


def slakh_category(inst_class: str, program: int) -> str:
    """Map a Slakh stem to a category. Unknown classes fall back to other_tonal."""
    if inst_class == "Piano" and program >= 4:
        return "keys"
    return _CLASS_TO_CAT.get(inst_class, "other_tonal")


def musdb_category(stem: str) -> str | None:
    """MUSDB stem name -> category, or None when the stem must not be used."""
    return {"vocals": "vocal_lead", "bass": "bass"}.get(stem)


# MUSDB18-HQ stream order inside each .stem.mp4 (SPEC F, measured on disk)
MUSDB_STREAMS = {"mix": 0, "drums": 1, "bass": 2, "other": 3, "vocals": 4}
