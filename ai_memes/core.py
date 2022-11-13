import logging
import os.path
from typing import Tuple

import click
import cv2
import numpy as np
import openai
from openai.error import RateLimitError
from retry import retry
from tqdm import tqdm
import itertools

from ai_memes.constants import (
    DATA_KEY_IN_API_RESPONSE,
    IMAGE_URL_KEY_IN_RESPONSE_DATA,
    EMPTY_TEXT,
)
from ai_memes.drawing import (
    put_text_on_image,
    generate_blank_image,
    concatenate_images_in_row,
    concatenate_images_rows,
)
from ai_memes.utils import download_image, slugify, split_list_into_batches

logging.getLogger().setLevel(logging.INFO)


@click.command()
@click.option("--title", type=str, required=True, help="Title of the meme.")
@click.option(
    "--image-description",
    "images_descriptions",
    type=str,
    required=True,
    multiple=True,
    help="Image description sent to Dall-E 2.",
)
@click.option(
    "--meme-text",
    "meme_texts",
    type=str,
    required=True,
    multiple=True,
    help="Text visible in the corresponding image of the meme.",
)
@click.option(
    "--target-directory", type=str, required=True, help="Directory to save results."
)
@click.option(
    "--columns-number",
    type=int,
    required=False,
    default=1,
    help="Number of columns for images in meme.",
)
@click.option(
    "--variants-number",
    type=int,
    required=False,
    default=1,
    help="Number of meme variants to be generated.",
)
@click.option(
    "--generated-images-size",
    type=(int, int),
    required=False,
    default=(512, 512),
    help="Size of each image in meme",
)
def generate_memes(
    title: str,
    images_descriptions: Tuple[str, ...],
    meme_texts: Tuple[str, ...],
    target_directory: str,
    columns_number: int,
    variants_number: int,
    generated_images_size: Tuple[int, int],
) -> None:
    os.makedirs(target_directory, exist_ok=True)
    title_slug = slugify(value=title)
    for variant_id in range(variants_number):
        target_path = os.path.abspath(
            os.path.join(target_directory, f"{title_slug}_{variant_id:03}.jpg")
        )
        meme_variant = generate_meme_variant(
            images_descriptions=images_descriptions,
            meme_texts=meme_texts,
            columns_number=columns_number,
            generated_images_size=generated_images_size,
        )
        logging.info(f"Saving generated meme into {target_path}")
        cv2.imwrite(target_path, meme_variant)


def generate_meme_variant(
    images_descriptions: Tuple[str, ...],
    meme_texts: Tuple[str, ...],
    columns_number: int,
    generated_images_size: Tuple[int, int],
) -> np.ndarray:
    generated_images = [
        generate_image(
            image_description=image_description,
            generated_images_size=generated_images_size,
        )
        for image_description in tqdm(
            images_descriptions, desc="Generating images using Dall-E 2..."
        )
    ]
    meme_images = []
    for image, meme_text in zip(itertools.cycle(generated_images), meme_texts):
        if meme_text.lower() != EMPTY_TEXT:
            meme_image = put_text_on_image(image=image, text=meme_text)
        else:
            meme_image = image
        meme_images.append(meme_image)
    missing_images = len(meme_images) % columns_number
    blank_images = [
        generate_blank_image(reference_image=meme_images[0])
    ] * missing_images
    meme_images.extend(blank_images)
    rows = []
    for row_images in split_list_into_batches(
        input_list=meme_images, batch_size=columns_number
    ):
        row = concatenate_images_in_row(images=row_images)
        rows.append(row)
    return concatenate_images_rows(rows=rows)


@retry(RateLimitError, tries=6, delay=60, logger=logging.getLogger())
def generate_image(
    image_description: str, generated_images_size: Tuple[int, int]
) -> np.ndarray:
    size = "x".join([str(e) for e in generated_images_size])
    response = openai.Image.create(
        prompt=image_description,
        n=1,
        size=size,
    )
    image_url = response[DATA_KEY_IN_API_RESPONSE][0][IMAGE_URL_KEY_IN_RESPONSE_DATA]
    return download_image(image_url=image_url)


if __name__ == "__main__":
    generate_memes()
