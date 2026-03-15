from __future__ import annotations

from math import cos, pi, sin

import numpy as np
from PIL import Image, ImageDraw

from .prompting import PromptProfile


SKIN_PALETTES = {
    "warm": (181, 120, 82),
    "medium": (120, 78, 50),
    "deep": (78, 48, 30),
}

BACKGROUND_PALETTES = {
    "warm": (217, 170, 112),
    "medium": (203, 120, 82),
    "deep": (114, 72, 48),
}


def _clamp_color(value: float) -> int:
    return max(0, min(255, int(value)))


def _tinted(color: tuple[int, int, int], scale: float) -> tuple[int, int, int]:
    return tuple(_clamp_color(channel * scale) for channel in color)


def _latent_value(latent: np.ndarray, row: int, col: int) -> float:
    return float(latent[row % latent.shape[0], col % latent.shape[1]])


def render_portrait(profile: PromptProfile, latent: np.ndarray, size: int = 512) -> Image.Image:
    canvas = Image.new("RGB", (size, size), BACKGROUND_PALETTES[profile.skin_tone])
    draw = ImageDraw.Draw(canvas)

    skin = SKIN_PALETTES[profile.skin_tone]
    shadow = _tinted(skin, 0.78)
    highlight = _tinted(skin, 1.12)
    hair_color = (
        _clamp_color(18 + abs(_latent_value(latent, 0, 0)) * 20),
        _clamp_color(14 + abs(_latent_value(latent, 0, 1)) * 16),
        _clamp_color(10 + abs(_latent_value(latent, 0, 2)) * 14),
    )

    face_width = size * (0.34 + 0.04 * _latent_value(latent, 1, 0))
    face_height = size * (0.46 + 0.03 * _latent_value(latent, 1, 1))
    cx, cy = size / 2, size * 0.54

    draw.rounded_rectangle(
        [
            cx - face_width / 2,
            cy - face_height / 2,
            cx + face_width / 2,
            cy + face_height / 2,
        ],
        radius=size * 0.1,
        fill=skin,
        outline=shadow,
        width=3,
    )

    hair_top = cy - face_height * 0.72
    hair_bottom = cy - face_height * 0.05
    hair_left = cx - face_width * 0.62
    hair_right = cx + face_width * 0.62

    if profile.hairstyle == "afro":
        for index in range(28):
            angle = 2 * pi * index / 28
            radius_x = face_width * 0.43 + 12 * _latent_value(latent, 0, 3)
            radius_y = face_height * 0.3 + 8 * _latent_value(latent, 1, 2)
            x = cx + cos(angle) * radius_x
            y = hair_bottom + sin(angle) * radius_y - face_height * 0.35
            draw.ellipse([x - 26, y - 26, x + 26, y + 26], fill=hair_color)
    elif profile.hairstyle == "braids":
        for offset in range(-4, 5):
            x = cx + offset * face_width * 0.1
            draw.line(
                [(x, hair_top), (x - 8 * sin(offset), cy + face_height * 0.22)],
                fill=hair_color,
                width=10,
            )
    elif profile.hairstyle == "locs":
        for offset in range(-5, 6):
            x = cx + offset * face_width * 0.09
            draw.rounded_rectangle(
                [x - 7, hair_top, x + 7, cy + face_height * 0.24],
                radius=8,
                fill=hair_color,
            )
    else:
        draw.ellipse(
            [hair_left, hair_top, hair_right, hair_bottom],
            fill=hair_color,
        )

    eye_y = cy - face_height * 0.08
    eye_dx = face_width * 0.2
    eye_w = face_width * 0.12
    eye_h = face_height * (0.04 + 0.01 * abs(_latent_value(latent, 2, 0)))
    for sign in (-1, 1):
        ex = cx + sign * eye_dx
        draw.ellipse([ex - eye_w, eye_y - eye_h, ex + eye_w, eye_y + eye_h], fill=(255, 255, 255))
        pupil_shift = 6 * _latent_value(latent, 2, 1)
        draw.ellipse(
            [ex - 8 + pupil_shift, eye_y - 8, ex + 8 + pupil_shift, eye_y + 8],
            fill=(35, 24, 18),
        )

    nose_top = eye_y + face_height * 0.06
    nose_bottom = cy + face_height * 0.12
    draw.polygon(
        [(cx, nose_top), (cx - 16, nose_bottom), (cx + 16, nose_bottom)],
        fill=highlight,
        outline=shadow,
    )

    mouth_y = cy + face_height * 0.23
    smile_factor = 0.0
    if profile.expression == "smile":
        smile_factor = 20 + 10 * _latent_value(latent, 3, 0)
    elif profile.expression == "calm":
        smile_factor = -8

    draw.arc(
        [cx - 50, mouth_y - 18 - smile_factor, cx + 50, mouth_y + 18 + smile_factor],
        start=10,
        end=170,
        fill=(92, 28, 32),
        width=4,
    )

    if profile.accessory == "glasses":
        lens_y = eye_y
        lens_w = 36
        lens_h = 26
        for sign in (-1, 1):
            lx = cx + sign * eye_dx
            draw.rounded_rectangle(
                [lx - lens_w, lens_y - lens_h, lx + lens_w, lens_y + lens_h],
                radius=8,
                outline=(40, 30, 24),
                width=4,
            )
        draw.line([(cx - eye_dx + lens_w, lens_y), (cx + eye_dx - lens_w, lens_y)], fill=(40, 30, 24), width=3)
    elif profile.accessory == "earrings":
        earring_y = cy + face_height * 0.1
        for sign in (-1, 1):
            ex = cx + sign * face_width * 0.42
            draw.ellipse([ex - 10, earring_y - 2, ex + 10, earring_y + 18], outline=(235, 198, 92), width=3)

    return canvas
