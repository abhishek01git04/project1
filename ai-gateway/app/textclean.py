"""Text normalisation shared by model training (training/) and the gateway at inference time."""

import re

_DATELINE = re.compile(r"^.{0,120}?\((?:reuters|ap|afp)\)\s*[-–—]\s*", re.I | re.S)
_AGENCY = re.compile(r"\b(?:reuters|getty images|featured image via)\b", re.I)
_URL = re.compile(r"https?://\S+|www\.\S+")
_WS = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Strip publisher fingerprints so models learn from content, not from who published it.

    Most real articles in WELFake come from Reuters and start with "CITY (Reuters) -".
    Left in, a model can score ~99% just by spotting the word "Reuters" and then fail
    on real-world text.
    """
    text = _DATELINE.sub("", text or "")
    text = _URL.sub(" ", text)
    text = _AGENCY.sub(" ", text)
    return _WS.sub(" ", text).strip()
