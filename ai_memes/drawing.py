import logging
from copy import copy
from typing import Tuple, List

import cv2
import numpy as np

from ai_memes.constants import (
    TEXT_LINE_CHARACTERS,
    FONT,
    MEME_FONT_SCALE_RATIO,
    TEXT_BORDER_THICKNESS,
    TEXT_COLOR,
    TEXT_BORDER_COLOR,
    TEXT_THICKNESS,
    TEXT_OY_CENTER,
    NATURAL_TEXT_SEPARATORS,
    PADDING_SIZE_RATIO,
    IMAGE_PADDING_COLOR,
    FOOTER_SIZE_RATIO,
)
from ai_memes.utils import flatten


def concatenate_images_in_row(images: List[np.ndarray]) -> np.ndarray:
    image_height = images[0].shape[0]
    image_width = images[0].shape[1]
    padding_size = int(round(PADDING_SIZE_RATIO * image_width, 0))
    vertical_padding = (
        np.ones((image_height, padding_size, 3)) * IMAGE_PADDING_COLOR
    ).astype(dtype=np.uint8)
    row_content = flatten([(vertical_padding, image) for image in images] + [(vertical_padding,)])  # type: ignore
    row = np.concatenate(row_content, axis=1)
    horizontal_padding = (
        np.ones((padding_size, row.shape[1], 3), dtype=np.uint8) * IMAGE_PADDING_COLOR
    )
    return np.concatenate([horizontal_padding, row], axis=0)


def concatenate_images_rows(rows: List[np.ndarray]) -> np.ndarray:
    footer_height = int(round(rows[0].shape[0] * FOOTER_SIZE_RATIO, 0))
    footer = (
        np.ones((footer_height, rows[0].shape[1], 3)) * IMAGE_PADDING_COLOR
    ).astype(dtype=np.uint8)
    font_scale = (rows[0].shape[0] * MEME_FONT_SCALE_RATIO) - 0.35
    center = footer.shape[1] // 2, footer.shape[0] // 2
    footer = put_text(
        image=footer,
        text="AI Meme | Pawel Peczek",
        font_scale=font_scale,
        center=center,
        color=TEXT_BORDER_COLOR,
    )
    rows_copy = copy(rows)
    rows_copy.append(footer)
    return np.concatenate(rows_copy, axis=0)


def put_text_on_image(image: np.ndarray, text: str) -> np.ndarray:
    text_lines = len(text) / TEXT_LINE_CHARACTERS
    if text_lines > 2:
        logging.warning("Text on the meme is to long and will be trimmed.")
        text = f"{text[:(2 * TEXT_LINE_CHARACTERS) - 3]}..."
    font_scale = image.shape[1] * MEME_FONT_SCALE_RATIO
    _, line_height = get_text_line_dimensions(
        text=text, font_scale=font_scale, font_thickness=TEXT_BORDER_THICKNESS
    )
    lines = [text[:TEXT_LINE_CHARACTERS], text[TEXT_LINE_CHARACTERS:]]
    # forgive me primitive operations on text :)
    if len(lines[1]) > 0 and lines[1][0] in NATURAL_TEXT_SEPARATORS:
        lines[0] = f"{lines[0]}{lines[1][0]}"
        lines[1] = lines[1][1:]
    if len(lines[1]) > 0 and lines[0][-1] not in NATURAL_TEXT_SEPARATORS:
        lines[0] = f"{lines[0]}-"
    text_center_x = image.shape[1] // 2
    if len(lines[1]) == 0:
        text_center_y = int(round(TEXT_OY_CENTER * image.shape[0]))
        return put_text(
            image=image,
            text=lines[0],
            font_scale=font_scale,
            center=(text_center_x, text_center_y),
        )
    oy_offset = line_height // 2 + 2
    text_center_y = int(round(TEXT_OY_CENTER * image.shape[0])) - oy_offset
    image = put_text(
        image=image,
        text=lines[0],
        font_scale=font_scale,
        center=(text_center_x, text_center_y),
    )
    text_center_y = int(round(TEXT_OY_CENTER * image.shape[0])) + oy_offset
    return put_text(
        image=image,
        text=lines[1],
        font_scale=font_scale,
        center=(text_center_x, text_center_y),
    )


def put_text(
    image: np.ndarray,
    text: str,
    font_scale: float,
    center: Tuple[int, int],
    color: Tuple[int, int, int] = TEXT_COLOR,
) -> np.ndarray:
    line_width, line_height = get_text_line_dimensions(
        text=text, font_scale=font_scale, font_thickness=TEXT_BORDER_THICKNESS
    )
    center_x, center_y = center
    text_x = center_x - (line_width // 2)
    text_y = center_y + (line_height // 2)
    if color != TEXT_BORDER_COLOR:
        image = cv2.putText(
            image,
            text,
            (text_x, text_y),
            FONT,
            font_scale,
            TEXT_BORDER_COLOR,
            TEXT_BORDER_THICKNESS,
        )
    return cv2.putText(
        image, text, (text_x + 1, text_y - 1), FONT, font_scale, color, TEXT_THICKNESS
    )


def get_text_line_dimensions(
    text: str, font_scale: float, font_thickness: int
) -> Tuple[int, int]:
    (line_width, line_height), _ = cv2.getTextSize(
        text, FONT, font_scale, font_thickness
    )
    return line_width, line_height


def generate_blank_image(reference_image: np.ndarray) -> np.ndarray:
    blank_image = np.ones_like(reference_image)
    return (blank_image * IMAGE_PADDING_COLOR).astype(dtype=np.uint8)
