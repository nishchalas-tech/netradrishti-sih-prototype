# NETRADRISHTI (नेत्रदृष्टि)
### Explainable AI for Diabetic Retinopathy Screening in Rural India
**Smart India Hackathon 2026 | Problem Statement: SIH26038 | Theme: Healthcare**

---

> [!IMPORTANT]
> **RESEARCH PROTOTYPE DISCLAIMER**
> NETRADRISHTI is an engineering and research prototype developed for Smart India Hackathon 2026.
> It is **NOT** a clinically certified or FDA/CDSCO-cleared medical diagnostic device.
> All prototype AI outputs represent computational risk stratification and must be verified by a licensed ophthalmologist.
> No clinical validation, empirical sensitivity/specificity, or real patient outcomes are claimed without multi-center clinical trials.

---

## 🌟 Executive Summary

Diabetic Retinopathy (DR) is the leading cause of preventable blindness among working-age adults in India. Over 77 million people live with diabetes in India, yet more than 70% of the rural population lacks access to trained ophthalmologists. 

**NETRADRISHTI** bridges this critical healthcare access gap by providing:
1. **Automated Image Quality Gate**: Prevents diagnostic errors by algorithmically assessing focus (Laplacian variance), illumination, and field-of-view before AI execution, prompting immediate recapture at the rural Primary Health Centre (PHC).
2. **Authentic Deep Learning Triage**: Employs a real trained PyTorch convolutional neural network (`NetraNet`) to classify scans into **NON-REFERABLE** and **REFERABLE** categories (extensible to the 5-tier International Clinical DR scale).
3. **Explainable AI (Grad-CAM)**: Generates gradient-weighted class activation heatmaps overlaying retinal anatomical structures, demystifying the "black box" for rural health workers and tele-ophthalmologists.
4. **Dual-Tier Doctor Priority Queue**: Automatically triages referable cases into a high-priority tele-ophthalmology queue while scheduling non-referable cases for routine annual monitoring.
5. **Discrete-Event Simulation (Simulink & Telemetry)**: Models queueing dynamics, camera acquisition bottlenecks, rural bandwidth constraints, and doctor review capacity to optimize resource deployment across rural districts.

---

## 🔄 End-to-End Clinical Workflow

```
Rural Clinic (PHC)
   │
   ├─► Patient Registration (ID, Age, Diabetes History)
   │
   ├─► Portable Fundus Camera / Image Upload
   │
   ├─► Automated Quality Gate
   │      ├─► [UNGRADABLE: Blur/Illumination/FOV Failure] ──► Recapture on-site (Blocks AI)
   │      └─► [IMAGE ACCEPTED: Focus & Contrast Verified]
   │
   ├─► Real AI Screening (NetraNet)
   │      ├─► Prediction: REFERABLE vs. NON-REFERABLE
   │      ├─► Authentic Softmax Confidence
   │      └─► Grad-CAM Spatial Heatmap & Overlay
   │
   ├─► Automated Priority Triage
   │      ├─► Referable DR  ──► 🔴 HIGH PRIORITY Doctor Queue
   │      └─► Non-Referable ──► 🔵 ROUTINE Follow-up Queue
   │
   ├─► Tele-Ophthalmologist Review (District/Tertiary Eye Hospital)
   │      ├─► Case Inspection (Fundus + Quality + Grad-CAM + Evidence)
   │      └─► Clinical Decision: [CONFIRM REFER] / [MARK FOR FOLLOW-UP] / [RECERTIFICATION]
   │
   └─► Deployment & Resource Simulation (Simulink / SimEvents)
          └─► Optimizes Cameras, AI Nodes, and Clinician Workload for Rural Districts
```

---

## 🖥️ Screen Architecture

### Screen 1: Rural Clinic Workstation
- Designed for Accredited Social Health Activists (ASHA) and rural multipurpose health workers.
- Patient registration form (ID, Age, symptoms, diabetic duration).
- Image acquisition interface supporting drag-and-drop file upload or instant Hackathon Demo Cases.
- **Automated Image Quality Gate**:
  - Focus / blur calculation using Laplacian variance $\sigma^2(\nabla^2 I_{green})$.
  - Illumination validation (mean luminance, underexposure, and corneal flash glare detection).
  - Field-of-View (FOV) verification via circular aperture segmentation.
  - If **UNGRADABLE**: Displays bold alert, specific clinical reason (*"Image too blurry"*, *"Poor illumination"*, etc.), and displays *"Please recapture the fundus image."* AI screening is strictly locked out.
  - If **ACCEPTED**: Displays green verification badge and unlocks the **RUN AI SCREENING** trigger.
- **AI Screening & Explainability**:
  - Executes real forward pass on `models/dr_model.pth`.
  - Computes and displays true model confidence (e.g. `99.3%`).
  - Displays original scan, standalone Grad-CAM heatmap, and blended spatial attribution overlay side-by-side.
  - Provides objective evidence text: *"Highlighted regions indicate spatial features that influenced the model prediction. Does not prove histological lesion presence."*

### Screen 2: Ophthalmologist / Doctor Review Dashboard
- Designed for certified ophthalmologists at district telemedicine hubs.
- **Priority Queues**:
  - **High Priority Queue**: Contains all referable cases requiring urgent specialist attention.
  - **Routine Queue**: Contains non-referable cases for standard periodic re-screening.
  - **Completed Reviews**: Audit trail of resolved cases.
- **Case Review Detail View**:
  - Patient demographics and rural clinic intake notes.
  - Fundus image and automated quality score report.
  - AI prototype classification and model confidence.
  - Grad-CAM activation heatmap overlay.
- **Doctor Clinical Decision Action Station**:
  - Clinical observation and prescription text input.
  - Action buttons:
    - `[ CONFIRM REFER ]`: Confirms urgent tertiary hospital referral for laser/anti-VEGF therapy.
    - `[ MARK FOR FOLLOW-UP ]`: Confirms 6–12 month routine rural monitoring.
    - `[ REQUEST RECERTIFICATION ]`: Flags borderline cases for dilated high-resolution re-imaging.
  - Stores doctor ID, decision, and audit timestamp.

### Screen 3: System / Simulation Dashboard (Simulink Model)
- Digital twin representing the companion MATLAB/Simulink SimEvents queueing model (`simulink/rural_dr_screening_model.m`).
- Models an 8-hour Primary Health Centre shift with parameters:
  - Patient arrival rate ($\lambda$ patients/hour)
  - Number of portable fundus cameras ($K$)
  - Quality gate rejection rate ($p_{reject}$) & on-site recapture loop
  - Rural telemedicine bandwidth ($Mbps$) & packet upload delays
  - Edge AI inference throughput ($\mu_{AI}$)
  - Tele-ophthalmologist review capacity ($\mu_{doc}$) & doctor count
- Interactive controls and telemetry charts:
  1. Patient arrivals vs. cumulative throughput
  2. Camera queue length and AI throughput
  3. Doctor priority queue buildup and average waiting time over the shift
  4. Resource utilization gauges (Cameras, AI server, Doctor duty)
  5. Operational bottleneck diagnosis (e.g. Camera limited vs. Doctor capacity limited).
- Exportable simulation output JSON file compatible with MATLAB/Simulink.

---

## ⚡ 60-Second Hackathon Judge Demo Walkthrough

The prototype includes three preloaded demo cases accessible from the sidebar:

1. **Case 001 (Poor Quality Image)**:
   - Click **`🔴 Case 001 (Poor Quality)`** in the sidebar.
   - Click **`CHECK IMAGE QUALITY`**.
   - **Result**: Displays **UNGRADABLE IMAGE** banner with reason *"Image too blurry"*. Shows *"Please recapture the fundus image."* AI screening button is safely blocked.
2. **Case 002 (Good Quality Non-Referable)**:
   - Click **`🟢 Case 002 (Non-Referable)`** in the sidebar.
   - Click **`CHECK IMAGE QUALITY`** -> Image Accepted.
   - Click **`RUN AI SCREENING`** -> AI classifies as **NON-REFERABLE** (Routine follow-up).
   - Shows Grad-CAM overlay and dispatches to **ROUTINE Doctor Queue**.
3. **Case 003 (Referable DR — High Priority)**:
   - Click **`🔥 Case 003 (Referable DR)`** in the sidebar.
   - Click **`CHECK IMAGE QUALITY`** -> Image Accepted.
   - Click **`RUN AI SCREENING`** -> AI classifies as **REFERABLE** with high confidence.
   - Shows Grad-CAM heatmap highlighting microaneurysms/hemorrhages.
   - Case is triaged to **HIGH PRIORITY Doctor Queue**.
   - Navigate to **`DOCTOR REVIEW`** tab -> Select Case 003 -> Click **`CONFIRM REFER`** -> Decision and timestamp recorded!
4. **Simulation Dashboard**:
   - Navigate to **`SIMULATION`** tab -> Adjust arrival rate or doctor count -> Click **`RUN SIMULATION`** -> Observe real-time queueing charts and bottleneck diagnosis.

---

## 📁 Repository Structure

```
netradrishti/
├── app.py                          # Streamlit Clinical Multi-Screen Application
├── config.py                       # Global thresholds, paths, and clinical constants
├── state_manager.py                # Session & disk store for patient triage queues and decisions
├── requirements.txt                # Python package dependencies
├── README.md                       # Comprehensive documentation
│
├── quality_gate/
│   ├── __init__.py
│   └── quality_checker.py         # Focus (Laplacian variance), illumination, and FOV checks
│
├── models/
│   ├── __init__.py
│   ├── dr_net.py                   # PyTorch NetraNet CNN architecture (extensible to ICDR 0-4)
│   └── dr_model.pth                # Real trained PyTorch model checkpoint
│
├── explainability/
│   ├── __init__.py
│   └── gradcam.py                  # Real Grad-CAM activation and gradient attribution hook
│
├── dataset/
│   ├── train/ (0_non_referable, 1_referable)
│   ├── validation/ (0_non_referable, 1_referable)
│   └── test/ (0_non_referable, 1_referable)
│
├── training/
│   ├── __init__.py
│   ├── prepare_sample_dataset.py   # Synthesizes anatomical fundus training/val/test images
│   └── train_model.py              # PyTorch training script (Adam, CrossEntropyLoss)
│
├── evaluation/
│   ├── __init__.py
│   └── metrics.py                  # Evaluates test set (accuracy, confusion matrix, ROC-AUC)
│
├── simulation/
│   ├── __init__.py
│   ├── simulink_engine.py          # Discrete-event queueing engine (SimEvents equivalent)
│   └── default_sim_output.json     # Baseline simulation data
│
├── matlab/
│   ├── preprocess_fundus.m         # MATLAB green channel and CLAHE preprocessing
│   ├── check_image_quality.m       # MATLAB Laplacian variance & quality assessment
│   └── infer_dr.m                  # MATLAB inference and Grad-CAM script
│
├── simulink/
│   ├── rural_dr_screening_model.m  # Simulink / SimEvents configuration and analytical M/M/c setup
│   └── README_SIMULINK.md          # Guide to running the Simulink model
│
└── demo_cases/
    ├── case001_ungradable/         # Sample ungradable (blurry) image
    ├── case002_non_referable/      # Sample normal fundus image
    └── case003_referable/          # Sample referable DR fundus image
```

---

## 🚀 Installation & Exact Run Commands

### Prerequisites
- Python 3.10+ (tested on Python 3.13)
- PyTorch 2.0+
- FastAPI & Uvicorn

### Step 1: Navigate to Project Directory
```bash
cd netradrishti-sih-prototype
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: (Optional) Re-evaluate Model on Test Cohort
```bash
python evaluation/metrics.py
```

### Step 4: Launch the NETRADRISHTI Clinical Workstation
```bash
python -m uvicorn server:app --host 127.0.0.1 --port 8000 --reload
```

Open your web browser and navigate to:
🌐 **`http://localhost:8000`**

---

## 🔌 RESTful API Reference

The enterprise FastAPI server provides clean, asynchronous REST endpoints:

| Endpoint | Method | Description |
| :--- | :---: | :--- |
| `/` | `GET` | Serves the formal clinical light web workstation |
| `/api/status` | `GET` | Health check & deep learning model load state (`ONLINE` / `MODEL NOT LOADED`) |
| `/api/demo-case/{case_id}` | `GET` | Retrieves preloaded demo cases (`case001`, `case002`, `case003`) |
| `/api/quality-check` | `POST` | Evaluates focus (Laplacian variance), illumination, and aperture coverage |
| `/api/run-screening` | `POST` | Executes NetraNet forward pass, authentic Grad-CAM overlay, and doctor queue dispatch |
| `/api/cases` | `GET` | Returns high-priority queue, routine queue, and complete patient directory |
| `/api/case/{case_id}` | `GET` | Returns single case report with Grad-CAM overlays for ophthalmologist review |
| `/api/doctor-decision` | `POST` | Records ophthalmologist clinical verdict, prescription, and audit timestamp |
| `/api/run-simulation` | `POST` | Executes discrete-event queueing simulation (Simulink equivalent) |
| `/api/simulation/default` | `GET` | Retrieves precomputed baseline simulation output |

---

## 🔬 Scientific & Architectural Rigor

1. **No Data Leakage**: The training, validation, and test splits are completely disjoint. Test images are never accessed during training or model checkpoint selection.
2. **Real Weights File**: The system loads `models/dr_model.pth`. If the file is deleted or missing, the system displays `"MODEL NOT LOADED"` and prohibits inference rather than hallucinating predictions.
3. **Real Softmax Confidence**: Confidence scores are raw mathematical probabilities computed directly by `torch.softmax(logits)`.
4. **Clinical Authority Preservation**: AI predictions are categorized strictly as decision support. Final decisions (`CONFIRM REFER`, `MARK FOR FOLLOW-UP`, `REQUEST RECERTIFICATION`) belong solely to the human clinician.
5. **Simulink Integration**: The discrete-event engine implements real $M/M/c$ queueing mathematics, providing actionable guidance to state health departments regarding camera-to-doctor ratios.

---

**Developed for Smart India Hackathon 2026**
*Advancing rural healthcare equity through Explainable AI.*

