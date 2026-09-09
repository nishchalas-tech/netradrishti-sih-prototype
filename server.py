"""
NETRADRISHTI — Enterprise FastAPI Application Server
Smart India Hackathon 2026 | Problem Statement: SIH26038
Explainable AI for Diabetic Retinopathy Screening in Rural India
"""

import os
import sys
import io
import time
import base64
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
import random
import numpy as np

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

# Add current directory to path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config import (
    RESEARCH_DISCLAIMER,
    PRIORITY_HIGH,
    PRIORITY_ROUTINE,
    DECISION_PENDING,
    DECISION_CONFIRM_REFER,
    DECISION_MARK_FOLLOWUP,
    DECISION_REQUEST_RECERT,
    DEMO_DIR,
    MODEL_WEIGHTS_PATH,
    CLASS_NAMES
)
from quality_gate.quality_checker import assess_image_quality
from quality_gate.enhancement import enhance_fundus_image
from models.dr_net import load_trained_model, run_inference, get_model_status
from explainability.gradcam import generate_cam_overlay
from state_manager import state_mgr
from simulation.simulink_engine import run_rural_deployment_simulation, load_simulation_results

# Initialize FastAPI App
app = FastAPI(
    title="NETRADRISHTI Clinical Tele-Triage API",
    description="Explainable AI for Diabetic Retinopathy Screening in Rural India (SIH26038)",
    version="2.1.0"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ICDR Level mapping (demo labels from binary classifier output)
ICDR_MAP = {
    "NON-REFERABLE": {"level": 0, "label": "DR Level 0 — No Apparent DR", "short": "ICDR 0"},
    "REFERABLE": {"level": 2, "label": "DR Level 2 — Moderate NPDR (Demo Label)", "short": "ICDR 2"}
}

# Helper function to convert PIL Image to base64 data URI
def pil_to_base64_data_uri(img: Image.Image, format: str = "PNG") -> str:
    buffered = io.BytesIO()
    img.save(buffered, format=format)
    img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/{format.lower()};base64,{img_b64}"

def base64_data_uri_to_pil(data_uri: str) -> Image.Image:
    if "," in data_uri:
        header, encoded = data_uri.split(",", 1)
    else:
        encoded = data_uri
    data = base64.b64decode(encoded)
    return Image.open(io.BytesIO(data)).convert("RGB")


# Preload initial demo cases into doctor queue if empty
if len(state_mgr.get_all_cases()) == 0:
    c2_path = DEMO_DIR / "case002_non_referable" / "fundus_case002.png"
    if c2_path.exists():
        state_mgr.add_case(
            case_id="CASE-DEMO-002",
            patient_id="PAT-RUR-3491",
            age=48,
            clinical_notes="Type 2 Diabetes (3 yrs). Routine annual eye checkup at Primary Health Centre.",
            image_path=str(c2_path),
            quality_info={
                "status": "IMAGE ACCEPTED",
                "quality_score": 78.5,
                "blur_score": 79.4,
                "illumination_score": 74.0,
                "fov_score": 66.6,
                "usable": True,
                "focus_status": "PASS",
                "illumination_status": "PASS",
                "fov_status": "PASS",
                "artifacts_status": "PASS"
            },
            ai_info={
                "prediction": "NON-REFERABLE",
                "confidence": 0.998,
                "evidence": "Regular retinal vasculature without detectable microaneurysms or hard exudates."
            }
        )

    c3_path = DEMO_DIR / "case003_referable" / "fundus_case003.png"
    if c3_path.exists():
        state_mgr.add_case(
            case_id="CASE-DEMO-003",
            patient_id="PAT-RUR-8812",
            age=61,
            clinical_notes="Type 2 Diabetes (12 yrs). Decreased visual acuity (6/18) and macular blurriness.",
            image_path=str(c3_path),
            quality_info={
                "status": "IMAGE ACCEPTED",
                "quality_score": 84.0,
                "blur_score": 1113.0,
                "illumination_score": 77.7,
                "fov_score": 66.6,
                "usable": True,
                "focus_status": "PASS",
                "illumination_status": "PASS",
                "fov_status": "PASS",
                "artifacts_status": "PASS"
            },
            ai_info={
                "prediction": "REFERABLE",
                "confidence": 0.999,
                "evidence": "Dense clusters of intraretinal hemorrhages and lipid exudates along temporal vascular arcade."
            }
        )


# -------------------------------------------------------------
# Pydantic Request Models
# -------------------------------------------------------------
class QualityCheckRequest(BaseModel):
    image_data: str  # Base64 Data URI

class EnhanceImageRequest(BaseModel):
    image_data: str  # Base64 Data URI

class ScreeningRequest(BaseModel):
    image_data: str  # Base64 Data URI
    patient_id: str
    age: int
    clinical_notes: Optional[str] = "Routine examination"
    screening_location: Optional[str] = "PHC / Rural Clinic"

class DecisionRequest(BaseModel):
    case_id: str
    decision: str
    notes: Optional[str] = ""
    doctor_id: Optional[str] = "Ophthalmologist Review (Tele-Medicine Desk)"

class SimulationRequest(BaseModel):
    arrival_rate: float = 18.0
    num_cameras: int = 2
    image_rejection_rate: float = 12.0
    bandwidth_mbps: float = 4.0
    ai_capacity_per_hour: float = 60.0
    doctor_capacity_per_hour: float = 15.0
    num_doctors: int = 1
    simulation_hours: float = 8.0
    referral_rate: float = 24.0


# -------------------------------------------------------------
# API Endpoints
# -------------------------------------------------------------
@app.get("/api/status")
def get_system_status():
    """Returns deep learning model status and system metadata."""
    is_ok, msg = get_model_status()
    return {
        "status": "ONLINE" if is_ok else "DEGRADED",
        "model_loaded": is_ok,
        "model_message": msg,
        "model_weights_path": str(MODEL_WEIGHTS_PATH),
        "architecture": "NetraNet (Deep CNN Backbone + Grad-CAM Hooks)",
        "classes": CLASS_NAMES,
        "disclaimer": RESEARCH_DISCLAIMER
    }


@app.get("/api/models-and-data")
def get_models_and_data():
    """Returns system architecture info, dataset metadata, and training info for the Architecture panel."""
    is_ok, msg = get_model_status()
    weights_path = MODEL_WEIGHTS_PATH
    weights_size_kb = 0
    if weights_path.exists():
        weights_size_kb = round(weights_path.stat().st_size / 1024, 1)

    # Dataset metadata from disk
    dataset_dir = BASE_DIR / "dataset"
    train_count = val_count = test_count = 0
    if dataset_dir.exists():
        for split in ["train", "validation", "test"]:
            split_dir = dataset_dir / split
            if split_dir.exists():
                imgs = list(split_dir.rglob("*.png")) + list(split_dir.rglob("*.jpg"))
                if split == "train":
                    train_count = len(imgs)
                elif split == "validation":
                    val_count = len(imgs)
                elif split == "test":
                    test_count = len(imgs)

    return {
        "model": {
            "name": "NetraNet",
            "type": "Convolutional Neural Network (CNN)",
            "architecture": "4 Conv Blocks + Batch Norm + Classifier Head",
            "gradcam_hook": "conv4[0] (final convolutional layer)",
            "input_size": "224 × 224 px (RGB)",
            "output_classes": CLASS_NAMES,
            "weights_file": weights_path.name,
            "weights_size_kb": weights_size_kb,
            "status": "LOADED" if is_ok else "NOT LOADED",
            "note": "Trained on synthetic fundus dataset for SIH2026 prototype demonstration."
        },
        "dataset": {
            "type": "Synthetic fundus images (research prototype)",
            "train_images": train_count,
            "val_images": val_count,
            "test_images": test_count,
            "total": train_count + val_count + test_count,
            "classes": CLASS_NAMES,
            "note": "Synthetic dataset generated for prototype. Real clinical deployment requires validated clinical dataset."
        },
        "pipeline": [
            "01 IMAGE ACQUISITION — Fundus Camera / DICOM Export",
            "02 QUALITY GATE — Focus (Laplacian), Illumination, FOV, Artifacts",
            "03 IMAGE ENHANCEMENT — CLAHE, Green-Channel Contrast, Unsharp Masking",
            "04 RETINAL ANALYSIS — NetraNet Forward Pass (PyTorch)",
            "05 DR GRADING — Binary: NON-REFERABLE / REFERABLE → ICDR Demo Label",
            "06 EXPLAINABILITY — Grad-CAM Spatial Attribution (conv4 hook)",
            "07 TRIAGE — High Priority / Routine Queue Assignment",
            "08 DOCTOR REVIEW — Tele-Ophthalmologist Clinical Decision",
            "09 RESOURCE SIMULATION — Discrete-Event Model (Python; MATLAB/Simulink integration pending)"
        ],
        "disclaimer": RESEARCH_DISCLAIMER
    }


@app.get("/api/demo-case/{case_id}")
def get_demo_case(case_id: str):
    """Provides curated demonstration cases for hackathon presentation."""
    case_map = {
        "case001": {
            "name": "Case 001 — Poor Quality Scan (Defocus Blur)",
            "patient_id": "PAT-DEMO-001",
            "age": 63,
            "clinical_notes": "Patient experienced optical tremor during capture. Suspected mild cataract.",
            "file": DEMO_DIR / "case001_ungradable" / "fundus_case001.png",
            "expected_outcome": "UNGRADABLE IMAGE (Blocks AI)",
            "quality_subchecks": {
                "focus": "FAIL",
                "illumination": "FAIL",
                "fov": "PASS",
                "artifacts": "PASS"
            }
        },
        "case002": {
            "name": "Case 002 — Normal Retinal Scan (Non-Referable)",
            "patient_id": "PAT-DEMO-002",
            "age": 46,
            "clinical_notes": "Type 2 Diabetes (4 yrs duration). Annual asymptomatic checkup.",
            "file": DEMO_DIR / "case002_non_referable" / "fundus_case002.png",
            "expected_outcome": "IMAGE ACCEPTED → NON-REFERABLE → Routine Queue",
            "quality_subchecks": {
                "focus": "PASS",
                "illumination": "PASS",
                "fov": "PASS",
                "artifacts": "PASS"
            }
        },
        "case003": {
            "name": "Case 003 — Diabetic Retinopathy Pathology (Referable)",
            "patient_id": "PAT-DEMO-003",
            "age": 59,
            "clinical_notes": "Type 2 Diabetes (11 yrs). Chronic floaters, central vision distortion.",
            "file": DEMO_DIR / "case003_referable" / "fundus_case003.png",
            "expected_outcome": "IMAGE ACCEPTED → REFERABLE DR → High Priority Queue",
            "quality_subchecks": {
                "focus": "PASS",
                "illumination": "PASS",
                "fov": "PASS",
                "artifacts": "PASS"
            }
        }
    }

    if case_id not in case_map:
        raise HTTPException(status_code=404, detail="Demo case not found.")

    meta = case_map[case_id]
    if not meta["file"].exists():
        raise HTTPException(status_code=500, detail="Demo case image missing on disk.")

    pil_img = Image.open(meta["file"])
    data_uri = pil_to_base64_data_uri(pil_img)

    return {
        "case_id": case_id,
        "name": meta["name"],
        "patient_id": meta["patient_id"],
        "age": meta["age"],
        "clinical_notes": meta["clinical_notes"],
        "expected_outcome": meta["expected_outcome"],
        "quality_subchecks": meta["quality_subchecks"],
        "image_data": data_uri
    }


@app.post("/api/quality-check")
def check_quality(req: QualityCheckRequest):
    """Executes automated focus, illumination, and field-of-view quality gate."""
    try:
        pil_img = base64_data_uri_to_pil(req.image_data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {e}")

    result = assess_image_quality(pil_img)

    return {
        "status": result.status,
        "usable": result.usable,
        "quality_score": result.quality_score,
        "blur_score": result.blur_score,
        "illumination_score": result.illumination_score,
        "fov_score": result.fov_score,
        "reason": result.reason,
        "recommendation": result.recommendation,
        "subchecks": {
            "focus": getattr(result, "focus_status", "N/A"),
            "illumination": getattr(result, "illumination_status", "N/A"),
            "fov": getattr(result, "fov_status", "N/A"),
            "artifacts": getattr(result, "artifacts_status", "N/A")
        }
    }


@app.post("/api/enhance-image")
def enhance_image(req: EnhanceImageRequest):
    """Applies CLAHE, green-channel enhancement, and unsharp masking. Returns enhanced image and preprocessing metrics."""
    try:
        pil_img = base64_data_uri_to_pil(req.image_data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {e}")

    try:
        enhanced_img, metrics = enhance_fundus_image(pil_img)
        enhanced_b64 = pil_to_base64_data_uri(enhanced_img)
        original_b64 = pil_to_base64_data_uri(pil_img)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Enhancement failed: {e}")

    return {
        "original_image": original_b64,
        "enhanced_image": enhanced_b64,
        "metrics": metrics,
        "label": "Preprocessing Demonstration — CLAHE + Green-Channel Enhancement + Unsharp Masking",
        "note": "Enhancement is applied for visual demonstration. AI inference uses the original uploaded image."
    }


@app.post("/api/run-screening")
def run_screening(req: ScreeningRequest):
    """Runs genuine model inference and Grad-CAM explainability, triaging into doctor queues."""
    is_ok, msg = get_model_status()
    if not is_ok:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="MODEL NOT LOADED: Genuine trained model weights not found. Fake predictions prohibited."
        )

    try:
        pil_img = base64_data_uri_to_pil(req.image_data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {e}")

    # Safety Gate Check: Prohibit inference on ungradable images
    q_result = assess_image_quality(pil_img)
    if not q_result.usable:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "UNGRADABLE IMAGE",
                "reason": q_result.reason,
                "recommendation": "Please recapture the fundus image. AI screening is locked for ungradable images."
            }
        )

    # 1. Forward Pass on Real NetraNet Model (with latency measurement)
    model, _ = load_trained_model()
    t_start = time.time()
    infer_output = run_inference(model, pil_img)
    latency_ms = round((time.time() - t_start) * 1000, 1)

    # 2. Authentic Grad-CAM Generation
    orig_img, hmap_img, overlay_img, evidence_text = generate_cam_overlay(
        model, pil_img, class_idx=infer_output["class_index"]
    )

    # Convert images to base64
    orig_b64 = pil_to_base64_data_uri(orig_img)
    heatmap_b64 = pil_to_base64_data_uri(hmap_img)
    overlay_b64 = pil_to_base64_data_uri(overlay_img)

    # 3. ICDR demo label mapping
    icdr_info = ICDR_MAP.get(infer_output["prediction"], {"level": -1, "label": "UNKNOWN", "short": "?"})

    # 4. Save Image & Triage into Doctor Priority Queue
    case_id = f"CASE-{datetime.now().strftime('%m%d')}-{np.random.randint(100, 999)}"
    saved_img_path = str(DEMO_DIR / f"capture_{case_id}.png")
    try:
        pil_img.save(saved_img_path)
    except Exception:
        pass

    case_record = state_mgr.add_case(
        case_id=case_id,
        patient_id=req.patient_id,
        age=req.age,
        clinical_notes=req.clinical_notes,
        image_path=saved_img_path,
        quality_info={
            "status": q_result.status,
            "quality_score": q_result.quality_score,
            "blur_score": q_result.blur_score,
            "illumination_score": q_result.illumination_score,
            "fov_score": q_result.fov_score,
            "usable": True,
            "focus_status": getattr(q_result, "focus_status", "PASS"),
            "illumination_status": getattr(q_result, "illumination_status", "PASS"),
            "fov_status": getattr(q_result, "fov_status", "PASS"),
            "artifacts_status": getattr(q_result, "artifacts_status", "PASS")
        },
        ai_info={
            "prediction": infer_output["prediction"],
            "confidence": infer_output["confidence"],
            "probabilities": infer_output["probabilities"],
            "evidence": evidence_text,
            "icdr_level": icdr_info["level"],
            "icdr_label": icdr_info["label"],
            "latency_ms": latency_ms
        }
    )

    return {
        "case_id": case_id,
        "prediction": infer_output["prediction"],
        "confidence": infer_output["confidence"],
        "probabilities": infer_output["probabilities"],
        "priority": case_record["priority"],
        "evidence_text": evidence_text,
        "icdr_level": icdr_info["level"],
        "icdr_label": icdr_info["label"],
        "icdr_short": icdr_info["short"],
        "latency_ms": latency_ms,
        "images": {
            "original": orig_b64,
            "heatmap": heatmap_b64,
            "overlay": overlay_b64
        },
        "quality": {
            "status": q_result.status,
            "quality_score": q_result.quality_score,
            "blur_score": q_result.blur_score,
            "illumination_score": q_result.illumination_score,
            "subchecks": {
                "focus": getattr(q_result, "focus_status", "PASS"),
                "illumination": getattr(q_result, "illumination_status", "PASS"),
                "fov": getattr(q_result, "fov_status", "PASS"),
                "artifacts": getattr(q_result, "artifacts_status", "PASS")
            }
        },
        "screening_location": req.screening_location or "PHC / Rural Clinic",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "triage_summary": f"Case successfully triaged into {case_record['priority']} Doctor Review Queue."
    }


@app.get("/api/cases")
def get_cases():
    """Retrieves all cases, high priority queue, and routine queue."""
    high_priority = state_mgr.get_high_priority_queue()
    routine = state_mgr.get_routine_queue()
    all_cases = state_mgr.get_all_cases()

    return {
        "total_count": len(all_cases),
        "high_priority_count": len(high_priority),
        "routine_count": len(routine),
        "high_priority_queue": high_priority,
        "routine_queue": routine,
        "all_cases": all_cases
    }


@app.get("/api/case/{case_id}")
def get_case_detail(case_id: str):
    """Retrieves single case record with base64 image data for doctor review."""
    case = state_mgr.get_case(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found.")

    img_b64 = None
    overlay_b64 = None
    heatmap_b64 = None
    if os.path.exists(case["image_path"]):
        try:
            pil_img = Image.open(case["image_path"])
            img_b64 = pil_to_base64_data_uri(pil_img)

            # Generate overlay on the fly
            is_ok, _ = get_model_status()
            if is_ok and case.get("ai_result"):
                model, _ = load_trained_model()
                c_idx = 1 if case["ai_result"].get("prediction") == "REFERABLE" else 0
                _, hmap_img, overlay_img, _ = generate_cam_overlay(model, pil_img, class_idx=c_idx)
                overlay_b64 = pil_to_base64_data_uri(overlay_img)
                heatmap_b64 = pil_to_base64_data_uri(hmap_img)
        except Exception:
            pass

    return {
        **case,
        "image_data": img_b64,
        "overlay_data": overlay_b64,
        "heatmap_data": heatmap_b64
    }


@app.post("/api/doctor-decision")
def record_decision(req: DecisionRequest):
    """Records ophthalmologist clinical decision, prescription, and audit timestamp."""
    updated = state_mgr.update_doctor_decision(
        case_id=req.case_id,
        decision=req.decision,
        notes=req.notes or "",
        doctor_id=req.doctor_id or "Ophthalmologist Review (Tele-Medicine Desk)"
    )
    if not updated:
        raise HTTPException(status_code=404, detail="Case ID not found.")

    return {
        "status": "SUCCESS",
        "case_id": req.case_id,
        "decision": req.decision,
        "doctor_review": updated["doctor_review"]
    }


@app.get("/api/report/{case_id}")
def download_case_report(case_id: str):
    """Generates and returns a PDF case screening report for the given case ID."""
    case = state_mgr.get_case(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found.")

    try:
        from evaluation.pdf_report import generate_pdf_report
        # Load images if available
        fundus_img = None
        overlay_img = None
        if os.path.exists(case.get("image_path", "")):
            try:
                fundus_img = Image.open(case["image_path"])
                is_ok, _ = get_model_status()
                if is_ok and case.get("ai_result"):
                    model, _ = load_trained_model()
                    c_idx = 1 if case["ai_result"].get("prediction") == "REFERABLE" else 0
                    _, _, ov_img, _ = generate_cam_overlay(model, fundus_img, class_idx=c_idx)
                    overlay_img = ov_img
            except Exception:
                pass

        pdf_bytes = generate_pdf_report(
            case_record=case,
            fundus_image=fundus_img,
            overlay_image=overlay_img
        )

        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename=NetraDrishti_{case_id}_report.pdf"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {e}")


@app.post("/api/run-simulation")
def run_simulation(req: SimulationRequest):
    """Executes discrete-event queuing simulation."""
    results = run_rural_deployment_simulation(
        arrival_rate=req.arrival_rate,
        num_cameras=req.num_cameras,
        image_rejection_rate=req.image_rejection_rate,
        bandwidth_mbps=req.bandwidth_mbps,
        ai_capacity_per_hour=req.ai_capacity_per_hour,
        doctor_capacity_per_hour=req.doctor_capacity_per_hour,
        num_doctors=req.num_doctors,
        simulation_hours=req.simulation_hours,
        referral_rate=req.referral_rate
    )

    # Append bottleneck recommendations
    kpis = results.get("kpis", {})
    recs = []
    if kpis.get("camera_utilization_pct", 0) > 85:
        recs.append("Camera utilization is high — consider adding a portable fundus camera unit.")
    if kpis.get("doctor_utilization_pct", 0) > 90:
        recs.append("Tele-ophthalmologist queue saturated — add a second duty doctor or extend shift hours.")
    if float(str(kpis.get("image_rejection_rate", "0")).replace("%", "")) > 20:
        recs.append("High image rejection rate — conduct PHC staff training on fundus capture technique.")
    if kpis.get("final_doctor_backlog", 0) > 10:
        recs.append("Doctor backlog is growing — AI triage prioritisation is essential to manage workload.")
    if not recs:
        recs.append("System operating within acceptable parameters for current workload scenario.")

    results["bottleneck_recommendations"] = recs
    results["simulation_demo_notice"] = (
        "SIMULATION DEMO — Python discrete-event model. Awaiting MATLAB/Simulink integration."
    )

    return results


@app.get("/api/simulation/default")
def get_default_simulation():
    data = load_simulation_results()
    if "bottleneck_recommendations" not in data:
        data["bottleneck_recommendations"] = [
            "Run the simulation with custom parameters to see bottleneck recommendations."
        ]
    data["simulation_demo_notice"] = (
        "SIMULATION DEMO — Python discrete-event model. Awaiting MATLAB/Simulink integration."
    )
    return data


# -------------------------------------------------------------
# Static Files & Frontend SPA Mount
# -------------------------------------------------------------
STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/")
def serve_index():
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        return JSONResponse({"error": "Frontend UI file missing. Initializing..."}, status_code=503)
    return FileResponse(str(index_file))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
