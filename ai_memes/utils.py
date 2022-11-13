import unicodedata
import re
from typing import Iterable, List, TypeVar, Generator

import cv2
import numpy as np
import requests

T = TypeVar("T")


def slugify(value: str, allow_unicode: bool = False) -> str:
    """
    Taken from https://github.com/django/django/blob/master/django/utils/text.py
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize("NFKC", value)
    else:
        value = (
            unicodedata.normalize("NFKD", value)
            .encode("ascii", "ignore")
            .decode("ascii")
        )
    value = re.sub(r"[^\w\s-]", "", value.lower())
    return re.sub(r"[-\s]+", "-", value).strip("-_")


def download_image(image_url: str) -> np.ndarray:
    response = requests.get(image_url)
    bytes_array = np.asarray(bytearray(response.content), dtype=np.uint8)
    return cv2.imdecode(bytes_array, cv2.IMREAD_COLOR)


def flatten(iterable: Iterable[Iterable[T]]) -> List[T]:
    return [element for nested in iterable for element in nested]


def split_list_into_batches(
    input_list: List[T], batch_size: int
) -> Generator[List[T], None, None]:
    for i in range(0, len(input_list), batch_size):
        yield input_list[i : i + batch_size]
