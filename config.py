"""
NETRADRISHTI - Explainable AI for Diabetic Retinopathy Screening in Rural India
Smart India Hackathon 2026 | Problem Statement: SIH26038
Global Configuration & System Thresholds
"""

from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
MODELS_DIR = BASE_DIR / "models"
DEMO_DIR = BASE_DIR / "demo_cases"
SIMULATION_DIR = BASE_DIR / "simulation"

# Model Paths & Architecture
MODEL_WEIGHTS_PATH = MODELS_DIR / "dr_model.pth"
IMAGE_SIZE = (224, 224)
NUM_CLASSES = 2
CLASS_NAMES = ["NON-REFERABLE", "REFERABLE"]

# Extensible International Clinical Diabetic Retinopathy (ICDR) scale definition
ICDR_SCALE = {
    0: "No Apparent DR (Non-Referable)",
    1: "Mild Non-Proliferative DR (Routine Follow-up)",
    2: "Moderate Non-Proliferative DR (Referable)",
    3: "Severe Non-Proliferative DR (Referable - Urgent)",
    4: "Proliferative DR (Referable - Immediate)"
}

# Image Quality Gate Thresholds
# Focus / Blur (Laplacian variance of grayscale image)
BLUR_THRESHOLD_MIN = 50.0

# Illumination (Mean pixel intensity in [0, 255])
ILLUMINATION_MIN = 40.0
ILLUMINATION_MAX = 225.0

# Illumination uniformity / histogram percentile checks
DARK_PIXEL_MAX_PERCENTAGE = 0.45   # > 45% near zero outside mask is unacceptable
BRIGHT_PIXEL_MAX_PERCENTAGE = 0.35 # > 35% saturated indicates severe flash glare

# Field of view / Retinal Mask Coverage
MIN_RETINAL_COVERAGE_PERCENTAGE = 45.0  # Percentage of image inside circular retinal aperture

# Priority levels for Triage
PRIORITY_HIGH = "HIGH PRIORITY"
PRIORITY_ROUTINE = "ROUTINE"

# Doctor Decisions
DECISION_PENDING = "PENDING REVIEW"
DECISION_CONFIRM_REFER = "CONFIRMED REFERRAL"
DECISION_MARK_FOLLOWUP = "ROUTINE FOLLOW-UP"
DECISION_REQUEST_RECERT = "REQUEST RECERTIFICATION"

# Medical Research Disclaimer
RESEARCH_DISCLAIMER = (
    "Research prototype — AI output requires ophthalmologist review. "
    "Not a clinically validated medical diagnostic system. All final clinical decisions belong to certified clinicians."
)
