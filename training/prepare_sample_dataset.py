"""
NETRADRISHTI Dataset Generation & Partitioning Script
Synthesizes anatomical retinal fundus representations for training, validation,
and testing splits, adhering to strict data isolation principles.
"""

import os
import sys
import math
import random
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DATASET_DIR, DEMO_DIR


def create_base_fundus(width=512, height=512, seed=None):
    """
    Synthesizes normal retinal fundus anatomy:
    - Circular retinal field with vignette
    - Retinal background gradient (choroidal orange-red)
    - Optic disc (yellowish-pink oval)
    - Macula (darker avascular zone)
    - Retinal vascular tree (arterioles and venules branching from optic nerve head)
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    img = Image.new("RGB", (width, height), (8, 8, 10))
    draw = ImageDraw.Draw(img)

    # Retinal Circular Boundary
    cx, cy = width // 2, height // 2
    radius = int(min(width, height) * 0.46)
    
    # Gradient background
    y, x = np.ogrid[:height, :width]
    dist_from_center = np.sqrt((x - cx)**2 + (y - cy)**2)
    fundus_mask = dist_from_center <= radius

    # Background pigment
    r_channel = np.clip(180 - (dist_from_center / radius * 50) + np.random.normal(0, 4, (height, width)), 0, 255)
    g_channel = np.clip(70 - (dist_from_center / radius * 35) + np.random.normal(0, 3, (height, width)), 0, 255)
    b_channel = np.clip(25 - (dist_from_center / radius * 15) + np.random.normal(0, 2, (height, width)), 0, 255)

    base_arr = np.zeros((height, width, 3), dtype=np.uint8)
    base_arr[:, :, 0] = np.where(fundus_mask, r_channel, 10)
    base_arr[:, :, 1] = np.where(fundus_mask, g_channel, 10)
    base_arr[:, :, 2] = np.where(fundus_mask, b_channel, 12)

    fundus_img = Image.fromarray(base_arr)
    draw = ImageDraw.Draw(fundus_img)

    # 1. Optic Disc (Nasal side, e.g. at x = cx - radius * 0.55)
    disc_x = int(cx - radius * 0.48 + random.randint(-10, 10))
    disc_y = int(cy + random.randint(-15, 15))
    disc_rx, disc_ry = int(radius * 0.16), int(radius * 0.18)

    draw.ellipse(
        [disc_x - disc_rx, disc_y - disc_ry, disc_x + disc_rx, disc_y + disc_ry],
        fill=(250, 225, 150)
    )
    # Physiologic cup inside disc
    draw.ellipse(
        [disc_x - int(disc_rx*0.6), disc_y - int(disc_ry*0.6), disc_x + int(disc_rx*0.6), disc_y + int(disc_ry*0.6)],
        fill=(255, 245, 200)
    )

    # 2. Macula / Fovea (Temporal side, darker pigment)
    macula_x = int(cx + radius * 0.22 + random.randint(-10, 10))
    macula_y = int(cy + random.randint(-8, 8))
    draw.ellipse(
        [macula_x - int(radius*0.14), macula_y - int(radius*0.14), macula_x + int(radius*0.14), macula_y + int(radius*0.14)],
        fill=(110, 35, 15)
    )

    # 3. Retinal Vascular Tree (Arterioles and Venules)
    # Major superior and inferior arcades curving around the macula
    def draw_vessel(start_pt, angles, lengths, widths, color):
        curr_x, curr_y = start_pt
        for ang, length, w in zip(angles, lengths, widths):
            rad = math.radians(ang)
            next_x = curr_x + length * math.cos(rad)
            next_y = curr_y + length * math.sin(rad)
            # Ensure inside fundus circle
            if np.sqrt((next_x - cx)**2 + (next_y - cy)**2) < radius * 0.95:
                draw.line([(curr_x, curr_y), (next_x, next_y)], fill=color, width=w)
            curr_x, curr_y = next_x, next_y

    vessel_color_venule = (100, 15, 15)
    vessel_color_artery = (135, 25, 20)

    # Superior temporal arcade
    draw_vessel(
        (disc_x, disc_y),
        [-75, -55, -35, -15, 5, 20],
        [40, 45, 45, 45, 40, 35],
        [5, 4, 3, 2, 2, 1],
        vessel_color_venule
    )
    # Inferior temporal arcade
    draw_vessel(
        (disc_x, disc_y),
        [75, 55, 35, 15, -5, -20],
        [40, 45, 45, 45, 40, 35],
        [5, 4, 3, 2, 2, 1],
        vessel_color_venule
    )
    # Superior nasal arcade
    draw_vessel(
        (disc_x, disc_y),
        [-110, -135, -150],
        [35, 40, 35],
        [4, 3, 2],
        vessel_color_artery
    )
    # Inferior nasal arcade
    draw_vessel(
        (disc_x, disc_y),
        [110, 135, 150],
        [35, 40, 35],
        [4, 3, 2],
        vessel_color_artery
    )

    # Smooth vessel edges slightly
    fundus_img = fundus_img.filter(ImageFilter.GaussianBlur(radius=0.7))
    return fundus_img, (cx, cy, radius, macula_x, macula_y)


def add_dr_pathology(img, fundus_meta, severity="moderate"):
    """
    Adds authentic hallmarks of Diabetic Retinopathy:
    - Microaneurysms: Tiny deep-red punctate dots (2-4 px)
    - Dot and Blot Hemorrhages: Irregular dark-red intraretinal lesions (5-12 px)
    - Hard Exudates: Sharp-bordered bright yellowish-white lipid deposits
    - Cotton Wool Spots: Fluffy white nerve fiber layer infarcts
    """
    cx, cy, radius, mx, my = fundus_meta
    draw = ImageDraw.Draw(img)

    num_lesions = 25 if severity == "moderate" else 55

    for _ in range(num_lesions):
        # Distribute lesions around posterior pole and vascular arcades
        angle = random.uniform(0, 2 * math.pi)
        r = random.uniform(radius * 0.15, radius * 0.75)
        lx = int(cx + r * math.cos(angle))
        ly = int(cy + r * math.sin(angle))

        lesion_type = random.choice(["microaneurysm", "hemorrhage", "exudate", "cotton_wool"])

        if lesion_type == "microaneurysm":
            rad = random.randint(2, 4)
            draw.ellipse([lx - rad, ly - rad, lx + rad, ly + rad], fill=(85, 10, 10))
        elif lesion_type == "hemorrhage":
            rad_x = random.randint(4, 9)
            rad_y = random.randint(3, 7)
            draw.ellipse([lx - rad_x, ly - rad_y, lx + rad_x, ly + rad_y], fill=(70, 8, 8))
        elif lesion_type == "exudate":
            # Cluster of bright yellowish lipid spots near macula
            rad = random.randint(2, 5)
            draw.ellipse([lx - rad, ly - rad, lx + rad, ly + rad], fill=(250, 245, 175))
        elif lesion_type == "cotton_wool":
            # Fluffy white lesion
            rad = random.randint(6, 12)
            draw.ellipse([lx - rad, ly - rad, lx + rad, ly + rad], fill=(235, 230, 220, 180))

    return img


def add_poor_quality_artifacts(img, artifact_type="blur"):
    """Applies camera / acquisition artifacts for Quality Gate testing."""
    if artifact_type == "blur":
        # Severe defocus blur
        return img.filter(ImageFilter.GaussianBlur(radius=8.5))
    elif artifact_type == "underexposed":
        # Very low illumination
        arr = np.array(img, dtype=np.float32) * 0.15
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    elif artifact_type == "overexposed":
        # Intense corneal flash reflection / glare
        arr = np.array(img, dtype=np.float32) * 1.8 + 80
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    elif artifact_type == "occluded":
        # Insufficient field of view / eyelid occlusion
        w, h = img.size
        draw = ImageDraw.Draw(img)
        draw.rectangle([0, 0, w, int(h * 0.65)], fill=(5, 5, 5))
        return img
    return img


def generate_dataset_and_demo_cases():
    """
    Builds the dataset splits and sample demo cases:
    - Train (50 non-referable, 50 referable)
    - Validation (15 non-referable, 15 referable)
    - Test (15 non-referable, 15 referable)
    - Demo Cases (Case 001, Case 002, Case 003)
    """
    splits = {
        "train": (40, 40),
        "validation": (12, 12),
        "test": (12, 12)
    }

    print("Generating NETRADRISHTI dataset splits...")

    seed_counter = 1000
    for split_name, (n_non_ref, n_ref) in splits.items():
        dir_non_ref = DATASET_DIR / split_name / "0_non_referable"
        dir_ref = DATASET_DIR / split_name / "1_referable"
        dir_non_ref.mkdir(parents=True, exist_ok=True)
        dir_ref.mkdir(parents=True, exist_ok=True)

        # Generate Non-referable
        for i in range(n_non_ref):
            seed_counter += 1
            fundus, _ = create_base_fundus(width=256, height=256, seed=seed_counter)
            out_path = dir_non_ref / f"img_nonref_{i+1:03d}.png"
            fundus.save(out_path)

        # Generate Referable
        for i in range(n_ref):
            seed_counter += 1
            fundus, meta = create_base_fundus(width=256, height=256, seed=seed_counter)
            fundus_dr = add_dr_pathology(fundus, meta, severity="moderate")
            out_path = dir_ref / f"img_ref_{i+1:03d}.png"
            fundus_dr.save(out_path)

        print(f"  [+] Split '{split_name}': {n_non_ref} non-referable, {n_ref} referable generated.")

    # Generate Standard Demo Cases
    print("Generating demo cases for presentation...")
    
    # Case 001: Poor quality image (Blurry / Underexposed)
    c1_dir = DEMO_DIR / "case001_ungradable"
    c1_dir.mkdir(parents=True, exist_ok=True)
    fundus, _ = create_base_fundus(width=384, height=384, seed=9901)
    fundus_bad = add_poor_quality_artifacts(fundus, artifact_type="blur")
    fundus_bad.save(c1_dir / "fundus_case001.png")

    # Case 002: Good quality Non-Referable image
    c2_dir = DEMO_DIR / "case002_non_referable"
    c2_dir.mkdir(parents=True, exist_ok=True)
    fundus_good_normal, _ = create_base_fundus(width=384, height=384, seed=9902)
    fundus_good_normal.save(c2_dir / "fundus_case002.png")

    # Case 003: Good quality Referable image (High Priority)
    c3_dir = DEMO_DIR / "case003_referable"
    c3_dir.mkdir(parents=True, exist_ok=True)
    fundus_ref_base, meta_ref = create_base_fundus(width=384, height=384, seed=9903)
    fundus_good_ref = add_dr_pathology(fundus_ref_base, meta_ref, severity="severe")
    fundus_good_ref.save(c3_dir / "fundus_case003.png")

    print("[SUCCESS] All dataset splits and demonstration cases generated successfully!")


if __name__ == "__main__":
    generate_dataset_and_demo_cases()
