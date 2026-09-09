"""
NETRADRISHTI — Explainable AI for Diabetic Retinopathy Screening in Rural India
Smart India Hackathon 2026 | Problem Statement: SIH26038
Research Prototype & Clinical Tele-Triage System
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import json
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt

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
    DEMO_DIR
)
from quality_gate.quality_checker import assess_image_quality, QualityResult
from models.dr_net import load_trained_model, run_inference, get_model_status
from explainability.gradcam import generate_cam_overlay
from state_manager import state_mgr
from simulation.simulink_engine import run_rural_deployment_simulation, load_simulation_results

# -------------------------------------------------------------
# Streamlit Page Configuration
# -------------------------------------------------------------
st.set_page_config(
    page_title="NETRADRISHTI | SIH 2026",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------------
# Custom Clinical UI Styling
# -------------------------------------------------------------
st.markdown("""
<style>
    /* Dark Slate & Clinical Teal Palette */
    .stApp {
        background-color: #0b1120;
        color: #f1f5f9;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    /* Top Header Banner */
    .netra-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f766e 100%);
        border-radius: 12px;
        padding: 20px 28px;
        margin-bottom: 24px;
        border: 1px solid #334155;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.5);
    }
    .netra-title {
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: 0.5px;
        color: #f8fafc;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .netra-subtitle {
        font-size: 0.95rem;
        color: #94a3b8;
        margin-top: 6px;
        margin-bottom: 0;
    }
    .badge-sih {
        background: #0ea5e9;
        color: #0f172a;
        font-weight: 700;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Card Styles */
    .clinical-card {
        background-color: #1e293b;
        border-radius: 10px;
        padding: 20px;
        border: 1px solid #334155;
        margin-bottom: 18px;
    }
    .card-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #e2e8f0;
        margin-bottom: 14px;
        display: flex;
        align-items: center;
        gap: 8px;
        border-bottom: 1px solid #334155;
        padding-bottom: 8px;
    }
    
    /* Quality Badges */
    .badge-accepted {
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid #10b981;
        padding: 6px 14px;
        border-radius: 6px;
        font-weight: 700;
        display: inline-block;
        font-size: 1rem;
    }
    .badge-ungradable {
        background: rgba(239, 68, 68, 0.15);
        color: #f87171;
        border: 1px solid #ef4444;
        padding: 6px 14px;
        border-radius: 6px;
        font-weight: 700;
        display: inline-block;
        font-size: 1rem;
    }
    
    /* Priority Badges */
    .priority-high {
        background: rgba(225, 29, 72, 0.2);
        color: #fb7185;
        border: 1px solid #e11d48;
        padding: 4px 10px;
        border-radius: 4px;
        font-weight: 700;
        font-size: 0.85rem;
    }
    .priority-routine {
        background: rgba(14, 165, 233, 0.2);
        color: #38bdf8;
        border: 1px solid #0284c7;
        padding: 4px 10px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    
    /* Safety Disclaimer Footer */
    .safety-footer {
        background: #0f172a;
        border-top: 1px solid #1e293b;
        color: #94a3b8;
        text-align: center;
        padding: 16px;
        margin-top: 40px;
        font-size: 0.85rem;
        border-radius: 8px;
    }
    
    /* Metric Card */
    .kpi-container {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 14px;
        text-align: center;
    }
    .kpi-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #38bdf8;
        margin: 4px 0;
    }
    .kpi-label {
        font-size: 0.8rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# Global State & Model Status Initialization
# -------------------------------------------------------------
model_ok, model_status_msg = get_model_status()

if "active_tab" not in st.session_state:
    st.session_state.active_tab = "RURAL CLINIC"

if "current_image" not in st.session_state:
    st.session_state.current_image = None

if "quality_result" not in st.session_state:
    st.session_state.quality_result = None

if "ai_result" not in st.session_state:
    st.session_state.ai_result = None

if "patient_id" not in st.session_state:
    st.session_state.patient_id = f"PAT-RUR-{np.random.randint(1000, 9999)}"

if "patient_age" not in st.session_state:
    st.session_state.patient_age = 54

if "patient_notes" not in st.session_state:
    st.session_state.patient_notes = "Type 2 Diabetes (6 yrs duration). Blurry vision reported in right eye."

if "demo_mode_case" not in st.session_state:
    st.session_state.demo_mode_case = None

# Preload initial demo cases into doctor queue if empty
if len(state_mgr.get_all_cases()) == 0:
    # Seed Demo Case 002 (Routine)
    state_mgr.add_case(
        case_id="CASE-DEMO-002",
        patient_id="PAT-RUR-3491",
        age=48,
        clinical_notes="Type 2 Diabetes (3 yrs). Routine annual eye screening.",
        image_path=str(DEMO_DIR / "case002_non_referable" / "fundus_case002.png"),
        quality_info={
            "status": "IMAGE ACCEPTED",
            "quality_score": 78.5,
            "blur_score": 82.1,
            "illumination_score": 74.0,
            "fov_score": 66.6,
            "usable": True
        },
        ai_info={
            "prediction": "NON-REFERABLE",
            "confidence": 0.998,
            "evidence": "Regular retinal vasculature without detectable microaneurysms or hard exudates."
        }
    )
    # Seed Demo Case 003 (Referable)
    state_mgr.add_case(
        case_id="CASE-DEMO-003",
        patient_id="PAT-RUR-8812",
        age=61,
        clinical_notes="Type 2 Diabetes (12 yrs). Reduced visual acuity (6/18).",
        image_path=str(DEMO_DIR / "case003_referable" / "fundus_case003.png"),
        quality_info={
            "status": "IMAGE ACCEPTED",
            "quality_score": 84.0,
            "blur_score": 1113.0,
            "illumination_score": 77.7,
            "fov_score": 66.6,
            "usable": True
        },
        ai_info={
            "prediction": "REFERABLE",
            "confidence": 0.999,
            "evidence": "Dense clusters of intraretinal hemorrhages and lipid exudates in macular region."
        }
    )

# -------------------------------------------------------------
# Top Navigation & Header
# -------------------------------------------------------------
st.markdown("""
<div class="netra-header">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
        <div>
            <div class="netra-title">
                <span>👁️ NETRADRISHTI</span>
                <span class="badge-sih">SIH 2026 • SIH26038</span>
            </div>
            <p class="netra-subtitle">Explainable AI for Diabetic Retinopathy Screening in Rural India | Research Prototype</p>
        </div>
        <div style="text-align: right; margin-top: 8px;">
            <span style="font-size: 0.8rem; color: #94a3b8;">Primary Health Centre (PHC) Tele-Triage</span><br/>
            <span style="color: #10b981; font-weight: 600; font-size: 0.85rem;">● SYSTEM ONLINE</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# Sidebar: System Controls & DEMO MODE
# -------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚡ HACKATHON DEMO MODE")
    st.caption("Select a curated clinical case to demonstrate the complete workflow in 60 seconds:")

    col_d1, col_d2 = st.columns(2)
    with col_d1:
        if st.button("🔴 Case 001\n(Poor Quality)", use_container_width=True):
            st.session_state.demo_mode_case = "case001"
            st.session_state.patient_id = "PAT-DEMO-001"
            st.session_state.patient_age = 63
            st.session_state.patient_notes = "Patient has slight cataract opacity and tremor during capture."
            img_p = DEMO_DIR / "case001_ungradable" / "fundus_case001.png"
            st.session_state.current_image = Image.open(img_p)
            st.session_state.quality_result = None
            st.session_state.ai_result = None
            st.session_state.active_tab = "RURAL CLINIC"
            st.rerun()

    with col_d2:
        if st.button("🟢 Case 002\n(Non-Referable)", use_container_width=True):
            st.session_state.demo_mode_case = "case002"
            st.session_state.patient_id = "PAT-DEMO-002"
            st.session_state.patient_age = 45
            st.session_state.patient_notes = "Asymptomatic; routine diabetic eye checkup."
            img_p = DEMO_DIR / "case002_non_referable" / "fundus_case002.png"
            st.session_state.current_image = Image.open(img_p)
            st.session_state.quality_result = None
            st.session_state.ai_result = None
            st.session_state.active_tab = "RURAL CLINIC"
            st.rerun()

    if st.button("🔥 Case 003 (Referable DR — High Priority)", use_container_width=True):
        st.session_state.demo_mode_case = "case003"
        st.session_state.patient_id = "PAT-DEMO-003"
        st.session_state.patient_age = 58
        st.session_state.patient_notes = "Long-standing diabetes with floaters and visual distortion."
        img_p = DEMO_DIR / "case003_referable" / "fundus_case003.png"
        st.session_state.current_image = Image.open(img_p)
        st.session_state.quality_result = None
        st.session_state.ai_result = None
        st.session_state.active_tab = "RURAL CLINIC"
        st.rerun()

    st.markdown("---")
    st.markdown("### ⚙️ Deep Learning Engine")
    if model_ok:
        st.success("✅ Real Model Loaded: `dr_model.pth`")
        st.caption("Architecture: `NetraNet` (CNN + Grad-CAM Hooks)")
    else:
        st.error(f"❌ {model_status_msg}")
        st.warning("Prediction blocked until genuine model weights are present.")

    st.markdown("---")
    st.markdown("### 📋 Navigation")
    selected_nav = st.radio(
        "Workflow View:",
        ["RURAL CLINIC", "DOCTOR REVIEW", "SIMULATION"],
        index=["RURAL CLINIC", "DOCTOR REVIEW", "SIMULATION"].index(st.session_state.active_tab)
    )
    if selected_nav != st.session_state.active_tab:
        st.session_state.active_tab = selected_nav
        st.rerun()

    st.markdown("---")
    st.caption("Smart India Hackathon 2026 • Theme: Healthcare")


# =============================================================
# SCREEN 1 — RURAL CLINIC WORKSTATION
# =============================================================
if st.session_state.active_tab == "RURAL CLINIC":
    st.markdown("## 🏥 Rural Clinic — Patient Screening & Quality Gate")
    st.caption("Primary Health Centre (PHC) Interface for Accredited Social Health Activists (ASHA) & Health Workers")

    # Patient Registration Section
    with st.container():
        st.markdown('<div class="clinical-card"><div class="card-title">1. Patient Registration</div>', unsafe_allow_html=True)
        col_p1, col_p2, col_p3 = st.columns([1.5, 1, 3])
        with col_p1:
            st.session_state.patient_id = st.text_input("Patient ID / Aadhaar Hash", value=st.session_state.patient_id)
        with col_p2:
            st.session_state.patient_age = st.number_input("Age (years)", min_value=12, max_value=110, value=int(st.session_state.patient_age))
        with col_p3:
            st.session_state.patient_notes = st.text_input("Clinical Notes / Symptoms", value=st.session_state.patient_notes)
        st.markdown('</div>', unsafe_allow_html=True)

    # Fundus Image Acquisition Section
    col_img_left, col_img_right = st.columns([1.2, 1.8])

    with col_img_left:
        st.markdown('<div class="clinical-card"><div class="card-title">2. Portable Camera / Image Upload</div>', unsafe_allow_html=True)
        
        if st.session_state.demo_mode_case:
            st.info(f"📌 **Demo Mode Active**: {st.session_state.demo_mode_case.upper()} loaded from test repository.")
            if st.button("Clear Demo & Upload New Image"):
                st.session_state.demo_mode_case = None
                st.session_state.current_image = None
                st.session_state.quality_result = None
                st.session_state.ai_result = None
                st.rerun()

        uploaded_file = st.file_uploader("Upload Retinal Fundus Image (.png, .jpg)", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            st.session_state.current_image = Image.open(uploaded_file)
            st.session_state.quality_result = None
            st.session_state.ai_result = None

        if st.session_state.current_image is not None:
            st.image(st.session_state.current_image, caption="Current Retinal Fundus Capture", use_container_width=True)
        else:
            st.warning("No fundus image loaded. Please upload a scan or click a Case in the Demo sidebar.")

        # Action: Check Image Quality Button
        btn_check_quality = st.button("🔍 CHECK IMAGE QUALITY", use_container_width=True, type="primary", disabled=(st.session_state.current_image is None))
        st.markdown('</div>', unsafe_allow_html=True)

    with col_img_right:
        st.markdown('<div class="clinical-card"><div class="card-title">3. Automated Image Quality Gate</div>', unsafe_allow_html=True)

        if btn_check_quality and st.session_state.current_image is not None:
            # Execute automated quality evaluation
            with st.spinner("Analyzing optical focus, illumination uniformity, and retinal field..."):
                st.session_state.quality_result = assess_image_quality(st.session_state.current_image)

        q_res = st.session_state.quality_result

        if q_res is not None:
            # Quality Gate Feedback
            if not q_res.usable:
                st.markdown(f"""
                <div style="background: rgba(239, 68, 68, 0.15); border: 2px solid #ef4444; border-radius: 8px; padding: 16px; margin-bottom: 16px;">
                    <div style="font-size: 1.3rem; font-weight: 800; color: #f87171;">⚠️ UNGRADABLE IMAGE</div>
                    <div style="font-size: 1.05rem; font-weight: 600; color: #fca5a5; margin-top: 6px;">
                        Rejection Reason: <span style="text-decoration: underline;">{q_res.reason}</span>
                    </div>
                    <div style="color: #f1f5f9; margin-top: 8px; font-weight: 700; font-size: 1.1rem;">
                        👉 Please recapture the fundus image.
                    </div>
                </div>
                """, unsafe_allow_html=True)

                col_q1, col_q2, col_q3 = st.columns(3)
                col_q1.metric("Quality Score", f"{q_res.quality_score} / 100", delta="-35.0", delta_color="inverse")
                col_q2.metric("Laplacian Focus", f"{q_res.blur_score:.1f}", "Threshold >= 50.0")
                col_q3.metric("Luminance", f"{q_res.illumination_score:.1f}", "40 - 225")

                st.error("⛔ SAFETY LOCK ACTIVE: AI Screening is BLOCKED for ungradable images to prevent spurious diagnoses.")
            else:
                st.markdown(f"""
                <div style="background: rgba(16, 185, 129, 0.15); border: 2px solid #10b981; border-radius: 8px; padding: 16px; margin-bottom: 16px;">
                    <div style="font-size: 1.3rem; font-weight: 800; color: #34d399;">✅ IMAGE ACCEPTED</div>
                    <div style="color: #d1fae5; margin-top: 6px; font-weight: 500;">
                        {q_res.recommendation}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                col_q1, col_q2, col_q3 = st.columns(3)
                col_q1.metric("Quality Score", f"{q_res.quality_score} / 100", "+Gradable")
                col_q2.metric("Laplacian Focus", f"{q_res.blur_score:.1f}", "Sharp")
                col_q3.metric("Luminance", f"{q_res.illumination_score:.1f}", "Optimum")

                # Allow AI Screening Button
                st.markdown("---")
                btn_run_ai = st.button("🚀 RUN AI SCREENING", type="primary", use_container_width=True)

                if btn_run_ai:
                    if not model_ok:
                        st.error("❌ MODEL NOT LOADED — Inference prohibited because trained weights are absent.")
                    else:
                        with st.spinner("Executing deep learning inference and computing Grad-CAM feature attribution..."):
                            model, _ = load_trained_model()
                            ai_eval = run_inference(model, st.session_state.current_image)
                            
                            # Generate Grad-CAM explainability
                            orig, hmap, overlay, evidence_desc = generate_cam_overlay(
                                model, st.session_state.current_image, class_idx=ai_eval["class_index"]
                            )

                            st.session_state.ai_result = {
                                **ai_eval,
                                "orig_img": orig,
                                "heatmap_img": hmap,
                                "overlay_img": overlay,
                                "evidence_desc": evidence_desc
                            }

                            # Triage into Doctor Queue
                            case_id = f"CASE-{datetime.now().strftime('%m%d')}-{np.random.randint(100, 999)}"
                            
                            # Save temp image for doctor inspection
                            case_img_path = str(DEMO_DIR / f"temp_{case_id}.png")
                            st.session_state.current_image.save(case_img_path)

                            state_mgr.add_case(
                                case_id=case_id,
                                patient_id=st.session_state.patient_id,
                                age=int(st.session_state.patient_age),
                                clinical_notes=st.session_state.patient_notes,
                                image_path=case_img_path,
                                quality_info={
                                    "status": q_res.status,
                                    "quality_score": q_res.quality_score,
                                    "blur_score": q_res.blur_score,
                                    "illumination_score": q_res.illumination_score,
                                    "fov_score": q_res.fov_score,
                                    "usable": True
                                },
                                ai_info={
                                    "prediction": ai_eval["prediction"],
                                    "confidence": ai_eval["confidence"],
                                    "evidence": evidence_desc
                                }
                            )
                            st.session_state.last_triaged_case = case_id
        else:
            st.info("Awaiting image quality assessment. Click **CHECK IMAGE QUALITY** on the left.")

        st.markdown('</div>', unsafe_allow_html=True)

    # AI SCREENING RESULTS & EXPLAINABILITY
    if st.session_state.ai_result is not None:
        ai_res = st.session_state.ai_result
        st.markdown('<div class="clinical-card"><div class="card-title">4. AI Screening Result & Explainability</div>', unsafe_allow_html=True)

        col_res1, col_res2 = st.columns([1.2, 2])
        
        with col_res1:
            if ai_res["prediction"] == "REFERABLE":
                st.markdown("""
                <div style="background: rgba(225, 29, 72, 0.2); border: 2px solid #e11d48; border-radius: 8px; padding: 18px; text-align: center;">
                    <div style="font-size: 0.85rem; text-transform: uppercase; color: #fb7185; font-weight: 700;">AI Triage Prediction</div>
                    <div style="font-size: 1.8rem; font-weight: 900; color: #fda4af; margin: 6px 0;">REFERABLE DR</div>
                    <div style="font-size: 0.95rem; color: #fecdd3;">High Risk of Retinopathy Detected</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="background: rgba(14, 165, 233, 0.2); border: 2px solid #0284c7; border-radius: 8px; padding: 18px; text-align: center;">
                    <div style="font-size: 0.85rem; text-transform: uppercase; color: #38bdf8; font-weight: 700;">AI Triage Prediction</div>
                    <div style="font-size: 1.8rem; font-weight: 900; color: #7dd3fc; margin: 6px 0;">NON-REFERABLE</div>
                    <div style="font-size: 0.95rem; color: #bae6fd;">Low Risk / Routine Follow-up Indicated</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br/>", unsafe_allow_html=True)
            st.metric("Prototype Model Confidence", f"{ai_res['confidence'] * 100:.1f}%")
            st.caption("Returned directly by the genuine PyTorch neural network softmax output.")

            # ICDR Scale Extensibility Note
            with st.expander("ℹ️ Clinical Scale Alignment"):
                st.caption(
                    "**Architecture Readiness:** Currently operational for binary screening (`NON-REFERABLE` vs `REFERABLE`). "
                    "Classifier head is architecturally prepared to map onto the 5-class International Clinical DR (ICDR) scale: "
                    "0 (No DR), 1 (Mild NPDR), 2 (Moderate NPDR), 3 (Severe NPDR), 4 (Proliferative DR)."
                )

        with col_res2:
            st.markdown("**Explainable AI (Grad-CAM Activation Visualizer)**")
            
            c_cam1, c_cam2, c_cam3 = st.columns(3)
            with c_cam1:
                st.image(ai_res["orig_img"], caption="1. Input Fundus", use_container_width=True)
            with c_cam2:
                st.image(ai_res["heatmap_img"], caption="2. Grad-CAM Heatmap", use_container_width=True)
            with c_cam3:
                st.image(ai_res["overlay_img"], caption="3. Spatial Attribution Overlay", use_container_width=True)

            st.markdown(f"""
            <div style="background: #0f172a; border-left: 4px solid #0ea5e9; padding: 10px 14px; border-radius: 4px; font-size: 0.9rem; margin-top: 10px;">
                <strong>Evidence Attribution:</strong> {ai_res['evidence_desc']}
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        # Automatic Queue Dispatch Confirmation
        if ai_res["prediction"] == "REFERABLE":
            st.warning(f"🚨 **Case Triaged**: Added to **HIGH PRIORITY Doctor Queue** for tele-ophthalmologist review.")
        else:
            st.success(f"📋 **Case Triaged**: Added to **ROUTINE Doctor Queue** for tele-ophthalmology verification.")

        if st.button("👉 View in Ophthalmologist Review Dashboard"):
            st.session_state.active_tab = "DOCTOR REVIEW"
            st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)


# =============================================================
# SCREEN 2 — OPHTHALMOLOGIST / DOCTOR REVIEW WORKSTATION
# =============================================================
elif st.session_state.active_tab == "DOCTOR REVIEW":
    st.markdown("## 👨‍⚕️ Ophthalmologist / Doctor Review Dashboard")
    st.caption("District & Tertiary Tele-Ophthalmology Workstation | Certified Clinician Decision Station")

    high_queue = state_mgr.get_high_priority_queue()
    routine_queue = state_mgr.get_routine_queue()
    all_cases = state_mgr.get_all_cases()

    # Queue Metrics Header
    col_qm1, col_qm2, col_qm3, col_qm4 = st.columns(4)
    with col_qm1:
        st.markdown(f"""
        <div class="kpi-container" style="border-left: 4px solid #ef4444;">
            <div class="kpi-value" style="color: #f87171;">{len(high_queue)}</div>
            <div class="kpi-label">🔴 High Priority (Referable)</div>
        </div>
        """, unsafe_allow_html=True)
    with col_qm2:
        st.markdown(f"""
        <div class="kpi-container" style="border-left: 4px solid #0284c7;">
            <div class="kpi-value" style="color: #38bdf8;">{len(routine_queue)}</div>
            <div class="kpi-label">🔵 Routine Follow-Up</div>
        </div>
        """, unsafe_allow_html=True)
    with col_qm3:
        reviewed_count = sum(1 for c in all_cases if c.get("doctor_review", {}).get("status") != DECISION_PENDING)
        st.markdown(f"""
        <div class="kpi-container" style="border-left: 4px solid #10b981;">
            <div class="kpi-value" style="color: #34d399;">{reviewed_count}</div>
            <div class="kpi-label">✅ Decisions Completed</div>
        </div>
        """, unsafe_allow_html=True)
    with col_qm4:
        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-value" style="color: #e2e8f0;">Dr. S. Sharma</div>
            <div class="kpi-label">Active Clinician (Reg: AIIMS-DL)</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)

    # Main Split: Queue Selection (Left) vs Detailed Case Review (Right)
    col_q_list, col_q_detail = st.columns([1.2, 2])

    with col_q_list:
        st.markdown('<div class="clinical-card"><div class="card-title">Priority Triage Queues</div>', unsafe_allow_html=True)
        
        queue_filter = st.radio(
            "Filter Queue View:",
            [f"High Priority ({len(high_queue)})", f"Routine Cases ({len(routine_queue)})", f"All Cases ({len(all_cases)})"],
            horizontal=True
        )

        if "High Priority" in queue_filter:
            display_cases = high_queue
        elif "Routine" in queue_filter:
            display_cases = routine_queue
        else:
            display_cases = all_cases

        if not display_cases:
            st.info("No cases currently in this queue.")
            selected_case_id = None
        else:
            case_options = [
                f"{c['case_id']} | {c['patient_id']} | {c['priority']} | {c['doctor_review']['status']}"
                for c in display_cases
            ]
            selected_option = st.selectbox("Select Case to Inspect:", case_options, index=0)
            selected_case_id = selected_option.split(" | ")[0]

        st.markdown('</div>', unsafe_allow_html=True)

    with col_q_detail:
        if selected_case_id:
            case_data = state_mgr.get_case(selected_case_id)
            if case_data:
                st.markdown(f'<div class="clinical-card"><div class="card-title">Case Review: {case_data["case_id"]} (Patient {case_data["patient_id"]})</div>', unsafe_allow_html=True)

                # Patient & Triage Header
                c_head1, c_head2, c_head3 = st.columns(3)
                c_head1.markdown(f"**Patient Age:** {case_data['age']} yrs")
                c_head2.markdown(f"**Screening Time:** `{case_data['created_at']}`")
                
                if case_data['priority'] == PRIORITY_HIGH:
                    c_head3.markdown('<span class="priority-high">🔴 HIGH PRIORITY</span>', unsafe_allow_html=True)
                else:
                    c_head3.markdown('<span class="priority-routine">🔵 ROUTINE</span>', unsafe_allow_html=True)

                st.markdown(f"**Rural Clinic Clinical Notes:** *{case_data['clinical_notes']}*")
                st.markdown("---")

                # Side-by-side Imaging & Evidence
                c_v1, c_v2 = st.columns([1, 1])
                
                with c_v1:
                    st.markdown("##### Retinal Fundus & Quality")
                    if os.path.exists(case_data['image_path']):
                        st.image(case_data['image_path'], caption=f"Fundus Image ({case_data['quality']['status']})", use_container_width=True)
                    else:
                        st.warning("Image file path unavailable.")

                    st.caption(f"**Quality Score:** {case_data['quality']['quality_score']}/100 | **Focus:** {case_data['quality']['blur_score']}")

                with c_v2:
                    st.markdown("##### AI Prediction & Grad-CAM Evidence")
                    st.markdown(f"**AI Prototype Classification:** `{case_data['ai_result']['prediction']}`")
                    st.markdown(f"**Model Softmax Confidence:** `{case_data['ai_result']['confidence']*100:.1f}%`")

                    # Generate live Grad-CAM if image available
                    if os.path.exists(case_data['image_path']) and model_ok:
                        case_img = Image.open(case_data['image_path'])
                        model, _ = load_trained_model()
                        c_idx = 1 if case_data['ai_result']['prediction'] == "REFERABLE" else 0
                        _, _, overlay_img, _ = generate_cam_overlay(model, case_img, class_idx=c_idx)
                        st.image(overlay_img, caption="Grad-CAM Activation Overlay", use_container_width=True)

                    st.markdown(f"<div style='font-size: 0.85rem; color: #94a3b8;'><em>{case_data['ai_result']['evidence']}</em></div>", unsafe_allow_html=True)

                st.markdown("---")
                
                # Clinical Decision Section
                st.markdown("#### ⚖️ Ophthalmologist Clinical Decision")
                st.markdown(
                    "<div style='color: #fbbf24; font-size: 0.85rem; margin-bottom: 12px;'>"
                    "<strong>Medical Authority Notice:</strong> The final diagnosis and management plan belongs strictly to the ophthalmologist. "
                    "AI outputs serve purely as triage and decision-support assistance."
                    "</div>",
                    unsafe_allow_html=True
                )

                current_status = case_data['doctor_review']['status']
                if current_status != DECISION_PENDING:
                    st.success(f"**Decision on Record:** `{current_status}` | Reviewed By: `{case_data['doctor_review']['doctor_id']}` at `{case_data['doctor_review']['timestamp']}`")
                    if case_data['doctor_review'].get('notes'):
                        st.info(f"**Doctor's Notes:** {case_data['doctor_review']['notes']}")

                doc_notes = st.text_area("Ophthalmologist Clinical Observations / Prescription / Referral Instructions:", value=case_data['doctor_review'].get('notes', ''))

                c_btn1, c_btn2, c_btn3 = st.columns(3)
                
                with c_btn1:
                    if st.button("✅ CONFIRM REFER", type="primary", use_container_width=True):
                        state_mgr.update_doctor_decision(
                            case_id=case_data['case_id'],
                            decision=DECISION_CONFIRM_REFER,
                            notes=doc_notes or "Urgent referral confirmed for tertiary diabetic retinopathy evaluation and management."
                        )
                        st.success("Decision recorded: CONFIRMED REFERRAL")
                        st.rerun()

                with c_btn2:
                    if st.button("📋 MARK FOR FOLLOW-UP", use_container_width=True):
                        state_mgr.update_doctor_decision(
                            case_id=case_data['case_id'],
                            decision=DECISION_MARK_FOLLOWUP,
                            notes=doc_notes or "Non-referable; recommended routine 12-month re-screening at PHC."
                        )
                        st.info("Decision recorded: ROUTINE FOLLOW-UP")
                        st.rerun()

                with c_btn3:
                    if st.button("🔄 RECERTIFICATION", use_container_width=True):
                        state_mgr.update_doctor_decision(
                            case_id=case_data['case_id'],
                            decision=DECISION_REQUEST_RECERT,
                            notes=doc_notes or "Clinical doubt / borderline image. Requesting dilated fundus imaging."
                        )
                        st.warning("Decision recorded: REQUEST RECERTIFICATION")
                        st.rerun()

                st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("Select a patient case from the left panel to review imaging and record clinical decisions.")


# =============================================================
# SCREEN 3 — SYSTEM / SIMULATION DASHBOARD (SIMULINK MODEL)
# =============================================================
elif st.session_state.active_tab == "SIMULATION":
    st.markdown("## 📊 Deployment & Resource Simulation Dashboard")
    st.caption("MATLAB / Simulink SimEvents Discrete-Event Model for Rural Tele-Ophthalmology Healthcare Optimization")

    st.markdown("""
    <div class="clinical-card">
        <div class="card-title">Simulink Discrete-Event Architecture</div>
        <div style="font-size: 0.9rem; color: #cbd5e1; line-height: 1.6;">
            This simulation implements the queueing dynamics of the companion <strong>Simulink SimEvents model</strong> 
            (<code>simulink/rural_dr_screening_model.m</code>). It simulates patient arrivals across rural Primary Health Centres (PHCs), 
            camera acquisition bottlenecks, image quality gate rejection & recapture loops, rural cellular bandwidth uplink constraints, 
            edge AI inference throughput, and tele-ophthalmologist review service times.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Simulation Controls
    with st.expander("🛠️ Configure Deployment & Clinical Capacity Parameters", expanded=True):
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            sim_arrival_rate = st.slider("Patient Arrival Rate (patients/hr)", min_value=5.0, max_value=40.0, value=18.0, step=1.0)
            sim_num_cameras = st.slider("Number of Portable Cameras", min_value=1, max_value=5, value=2)
        with col_s2:
            sim_bandwidth = st.slider("Telemedicine Bandwidth (Mbps)", min_value=0.5, max_value=25.0, value=4.0, step=0.5)
            sim_ai_capacity = st.slider("AI Edge Throughput (cases/hr)", min_value=20.0, max_value=120.0, value=60.0, step=5.0)
        with col_s3:
            sim_num_doctors = st.slider("Tele-Ophthalmologists on Duty", min_value=1, max_value=4, value=1)
            sim_doc_capacity = st.slider("Doctor Review Speed (cases/hr/doc)", min_value=5.0, max_value=30.0, value=15.0, step=1.0)

        col_s4, col_s5 = st.columns([1, 1])
        with col_s4:
            sim_rejection_rate = st.slider("Quality Gate Rejection Rate (%)", min_value=2.0, max_value=30.0, value=12.0, step=1.0)
        with col_s5:
            sim_duration_hours = st.slider("Shift Duration (hours)", min_value=4.0, max_value=12.0, value=8.0, step=1.0)

        btn_run_sim = st.button("▶️ RUN SIMULATION", type="primary", use_container_width=True)

    # Run or load simulation
    if btn_run_sim:
        with st.spinner("Executing discrete-event queuing simulation over shift..."):
            sim_data = run_rural_deployment_simulation(
                arrival_rate=sim_arrival_rate,
                num_cameras=sim_num_cameras,
                image_rejection_rate=sim_rejection_rate,
                bandwidth_mbps=sim_bandwidth,
                ai_capacity_per_hour=sim_ai_capacity,
                doctor_capacity_per_hour=sim_doc_capacity,
                num_doctors=sim_num_doctors,
                simulation_hours=sim_duration_hours
            )
            st.session_state.current_sim = sim_data
    elif "current_sim" not in st.session_state:
        st.session_state.current_sim = load_simulation_results()

    sim_data = st.session_state.current_sim
    kpis = sim_data["kpis"]

    # KPI Telemetry Cards
    st.markdown("### 📈 Clinical Operations Telemetry")
    col_k1, col_k2, col_k3, col_k4, col_k5 = st.columns(5)
    
    col_k1.markdown(f"""
    <div class="kpi-container">
        <div class="kpi-value">{kpis['total_patients_arrived']}</div>
        <div class="kpi-label">Total Patient Arrivals</div>
    </div>
    """, unsafe_allow_html=True)

    col_k2.markdown(f"""
    <div class="kpi-container">
        <div class="kpi-value">{kpis['images_captured_per_hour']}</div>
        <div class="kpi-label">Images / Hour</div>
    </div>
    """, unsafe_allow_html=True)

    col_k3.markdown(f"""
    <div class="kpi-container">
        <div class="kpi-value" style="color: #fbbf24;">{kpis['image_rejection_rate']}</div>
        <div class="kpi-label">Rejection Rate</div>
    </div>
    """, unsafe_allow_html=True)

    col_k4.markdown(f"""
    <div class="kpi-container">
        <div class="kpi-value" style="color: #34d399;">{kpis['cases_reviewed_by_doctor']}</div>
        <div class="kpi-label">Doctor Reviews Completed</div>
    </div>
    """, unsafe_allow_html=True)

    col_k5.markdown(f"""
    <div class="kpi-container">
        <div class="kpi-value">{kpis['average_waiting_time_min']} m</div>
        <div class="kpi-label">Avg Waiting Time</div>
    </div>
    """, unsafe_allow_html=True)

    # Bottleneck Analysis Banner
    st.markdown("<br/>", unsafe_allow_html=True)
    if "bottleneck" in kpis.get("bottleneck_diagnosis", "").lower() and "no critical" not in kpis.get("bottleneck_diagnosis", "").lower():
        st.warning(f"⚠️ **Operational Bottleneck Detected:** {kpis['bottleneck_diagnosis']}")
    else:
        st.success(f"✅ **System Status:** {kpis['bottleneck_diagnosis']}")

    # Interactive Telemetry Charts
    ts = sim_data["time_series"]
    time_hrs = ts["time_steps_hours"]

    col_ch1, col_ch2 = st.columns(2)

    with col_ch1:
        st.markdown("#### 1. Patient Arrivals vs AI Throughput")
        chart_df1 = pd.DataFrame({
            "Time (Hours)": time_hrs,
            "Cumulative Arrivals": ts["cumulative_arrivals"],
            "Camera Queue Length": ts["camera_queue_length"]
        }).set_index("Time (Hours)")
        st.line_chart(chart_df1, color=["#38bdf8", "#fbbf24"])

    with col_ch2:
        st.markdown("#### 2. Doctor Priority Queue & Waiting Time")
        chart_df2 = pd.DataFrame({
            "Time (Hours)": time_hrs,
            "Doctor Priority Queue": ts["doctor_priority_queue"],
            "Avg Wait Time (min)": ts["waiting_time_minutes"]
        }).set_index("Time (Hours)")
        st.line_chart(chart_df2, color=["#f43f5e", "#a855f7"])

    # Resource Utilization Chart
    st.markdown("#### 3. Resource Utilization Breakdown")
    col_u1, col_u2, col_u3 = st.columns(3)
    col_u1.progress(kpis["camera_utilization_pct"] / 100.0, text=f"Portable Cameras: {kpis['camera_utilization_pct']}%")
    col_u2.progress(kpis["ai_utilization_pct"] / 100.0, text=f"Edge AI Inference Unit: {kpis['ai_utilization_pct']}%")
    col_u3.progress(kpis["doctor_utilization_pct"] / 100.0, text=f"Tele-Ophthalmologist Duty: {kpis['doctor_utilization_pct']}%")

    # Export Section
    st.markdown("---")
    st.markdown("#### 💾 Export & MATLAB/Simulink Integration")
    col_exp1, col_exp2 = st.columns([1.5, 2])
    with col_exp1:
        sim_json_str = json.dumps(sim_data, indent=2)
        st.download_button(
            "📥 Download Simulation Output (.JSON)",
            data=sim_json_str,
            file_name="netradrishti_simulink_simulation_output.json",
            mime="application/json",
            use_container_width=True
        )
    with col_exp2:
        st.caption(
            "This JSON payload can be loaded directly by MATLAB using `jsondecode(fileread(...))` "
            "or compared against live SimEvents scopes in `simulink/rural_dr_screening_model.m`."
        )


# -------------------------------------------------------------
# Safety & Regulatory Disclaimer Footer
# -------------------------------------------------------------
st.markdown(f"""
<div class="safety-footer">
    <strong>NETRADRISHTI — Smart India Hackathon 2026 (SIH26038)</strong><br/>
    {RESEARCH_DISCLAIMER}
</div>
""", unsafe_allow_html=True)
