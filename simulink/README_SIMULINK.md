# NETRADRISHTI — Simulink / SimEvents Rural Screening Model

## Overview
This directory contains the MATLAB and Simulink SimEvents model architecture for simulating rural diabetic retinopathy screening deployments across Primary Health Centres (PHCs) in India.

Problem Statement: **SIH26038** (Smart India Hackathon 2026)

---

## Architecture of the Simulink Model

The discrete-event simulation model mirrors the clinical workflow:

```
[Entity Generator] Patient Arrivals (Poisson Lambda)
        │
        ▼
[Entity Queue] Clinic Waiting Area (FIFO)
        │
        ▼
[Entity Server] Portable Fundus Camera Acquisition (K Cameras)
        │
        ▼
[Output Switch] Quality Gate (Laplacian Variance / Illumination)
        ├──────► [Rejection/Recapture Loop] (12% rejection, 75% on-site recapture)
        │
        ▼
[Entity Server] Edge AI Inference Node (NetraNet + Grad-CAM, ~1 min/patient)
        │
        ▼
[Output Switch] AI Triage Filter
        ├──────► Non-Referable: Routine Follow-up Report (76%)
        │
        ▼
[Entity Queue] Tele-Ophthalmologist Priority Queue
        │
        ▼
[Entity Server] Doctor Review Station (M/M/c Queue, Doctor Capacity)
        │
        ▼
[Entity Sink] Clinical Decision Confirmed & Patient Discharge
```

---

## Files

1. **`rural_dr_screening_model.m`**:
   - Initializes all clinical parameters, queue capacities, arrival rates, and service distributions.
   - Computes analytical $M/M/c$ queueing bounds (utilization factor $\rho$, bottleneck identification).
   - Generates the JSON bridge `simulation/simulink_config_export.json` connecting directly with the NETRADRISHTI simulation dashboard.

2. **`../simulation/simulink_engine.py`**:
   - High-performance Python discrete-event engine reproducing the identical SimEvents state transitions for immediate live execution in the NETRADRISHTI web application.

3. **`../matlab/`**:
   - `preprocess_fundus.m`: Clinical green-channel extraction and CLAHE enhancement.
   - `check_image_quality.m`: Laplacian variance, illumination, and FOV checks.
   - `infer_dr.m`: Model inference and Grad-CAM feature attribution.

---

## How to Run in MATLAB / Simulink

1. Open MATLAB R2022b or later with the **SimEvents** and **Deep Learning Toolbox** installed.
2. Navigate to the `simulink/` folder:
   ```matlab
   cd('path/to/netradrishti/simulink');
   ```
3. Run the configuration script:
   ```matlab
   rural_dr_screening_model
   ```
4. To modify deployment parameters (e.g. increase doctors from 1 to 2, or test high patient surge):
   - Edit `NumCameras` or `NumOphthalmologists` in `rural_dr_screening_model.m`.
   - Re-run the script.
