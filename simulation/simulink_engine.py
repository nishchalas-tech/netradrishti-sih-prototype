"""
NETRADRISHTI Deployment & Resource Simulation Engine
Simulates the discrete-event rural screening queueing system corresponding to
the MATLAB/Simulink SimEvents model:
Patient Arrivals -> Camera Slots -> Quality Gate (Reject/Recapture) ->
Edge AI Inference -> Triage -> Doctor Priority Queue -> Ophthalmologist Review.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np

from config import SIMULATION_DIR

DEFAULT_SIM_OUTPUT_PATH = SIMULATION_DIR / "default_sim_output.json"


def run_rural_deployment_simulation(
    arrival_rate: float = 18.0,          # Patients arriving per hour
    num_cameras: int = 2,               # Number of portable fundus cameras in clinic
    image_rejection_rate: float = 12.0,  # % of images rejected by quality gate (blur/exposure)
    bandwidth_mbps: float = 4.0,        # Network bandwidth (Mbps) to district hospital
    ai_capacity_per_hour: float = 60.0, # AI inference throughput (cases/hour)
    doctor_capacity_per_hour: float = 15.0, # Reviews per doctor per hour
    num_doctors: int = 1,               # Number of tele-ophthalmologists on duty
    simulation_hours: float = 8.0,      # Clinical shift duration (e.g. 8 hours)
    referral_rate: float = 24.0,        # % of gradable images screened as Referable DR
    random_seed: int = 42
) -> Dict[str, Any]:
    """
    Executes discrete-event queuing simulation over the specified shift duration.
    Calculates dynamic queue lengths, waiting times, and resource utilizations.
    """
    np.random.seed(random_seed)

    # Convert shift to minutes
    total_minutes = int(simulation_hours * 60)
    time_steps = np.arange(0, total_minutes + 1, 15) # 15-minute time steps

    # Parameters per minute
    lambda_per_min = arrival_rate / 60.0
    cam_service_time_mean = 4.5 # minutes per patient acquisition
    ai_service_time = 60.0 / ai_capacity_per_hour # minutes per AI case
    doc_service_rate_per_min = (doctor_capacity_per_hour * num_doctors) / 60.0

    # Image upload transfer time (compressed fundus package ~ 2.5 MB)
    image_size_mb = 2.5
    transfer_time_min = (image_size_mb * 8) / (bandwidth_mbps * 60) if bandwidth_mbps > 0 else 0.05

    # State tracking
    total_arrivals = 0
    total_captured = 0
    total_rejected = 0
    total_recaptured = 0
    total_gradable = 0
    total_referable = 0
    total_non_referable = 0
    total_doctor_reviewed = 0

    # Queues
    camera_queue_history = []
    doctor_queue_history = []
    wait_time_history = []
    ai_throughput_history = []
    doc_utilization_history = []
    cumulative_arrivals_history = []

    current_camera_queue = 0
    current_doc_queue = 0
    busy_cameras = 0
    cam_busy_times = [0.0] * num_cameras

    # Simulation loop over 15-min intervals
    interval_arrivals = 0
    interval_ai_done = 0
    interval_doc_done = 0

    for minute in range(total_minutes + 1):
        # 1. Patient Arrivals (Poisson)
        arrivals_now = np.random.poisson(lambda_per_min)
        total_arrivals += arrivals_now
        current_camera_queue += arrivals_now
        interval_arrivals += arrivals_now

        # 2. Camera Acquisition Service
        for c in range(num_cameras):
            if cam_busy_times[c] > 0:
                cam_busy_times[c] -= 1.0
            elif current_camera_queue > 0:
                current_camera_queue -= 1
                total_captured += 1
                # Duration for fundus imaging
                cam_busy_times[c] = max(2.0, np.random.normal(cam_service_time_mean, 0.8))

                # Quality Gate Check
                is_rejected = (np.random.uniform(0, 100) < image_rejection_rate)
                if is_rejected:
                    total_rejected += 1
                    # 75% of ungradable cases succeed on immediate recapture
                    if np.random.uniform(0, 100) < 75.0:
                        total_recaptured += 1
                        total_gradable += 1
                        # Recapture adds 2 minutes to camera occupancy
                        cam_busy_times[c] += 2.0
                        gradable_case = True
                    else:
                        gradable_case = False
                else:
                    total_gradable += 1
                    gradable_case = True

                # If gradable, proceed to AI & Telemedicine
                if gradable_case:
                    interval_ai_done += 1
                    # Triage: Referable vs Non-Referable
                    is_referable = (np.random.uniform(0, 100) < referral_rate)
                    if is_referable:
                        total_referable += 1
                        current_doc_queue += 1
                    else:
                        total_non_referable += 1

        # 3. Tele-Ophthalmologist Review Service
        doc_capacity_this_min = np.random.poisson(doc_service_rate_per_min)
        doc_served = min(current_doc_queue, doc_capacity_this_min)
        current_doc_queue -= doc_served
        total_doctor_reviewed += doc_served
        interval_doc_done += doc_served

        # 4. Telemetry sampling every 15 minutes
        if minute % 15 == 0:
            active_cams = sum(1 for t in cam_busy_times if t > 0)
            cam_util = (active_cams / num_cameras) * 100.0 if num_cameras > 0 else 0
            
            # Estimated waiting time for a referable patient (Camera wait + AI + Doctor queue wait)
            avg_cam_wait = (current_camera_queue * cam_service_time_mean) / max(1, num_cameras)
            avg_doc_wait = (current_doc_queue / max(1e-3, doctor_capacity_per_hour * num_doctors)) * 60.0
            est_wait = round(avg_cam_wait + transfer_time_min + ai_service_time + avg_doc_wait, 1)

            # Instantaneous doctor utilization
            doc_util = min(100.0, (current_doc_queue / max(1, num_doctors * 4)) * 100.0) if current_doc_queue > 0 else 0.0

            camera_queue_history.append(int(current_camera_queue))
            doctor_queue_history.append(int(current_doc_queue))
            wait_time_history.append(float(est_wait))
            ai_throughput_history.append(int(interval_ai_done))
            doc_utilization_history.append(round(doc_util, 1))
            cumulative_arrivals_history.append(int(total_arrivals))

            interval_ai_done = 0

    # Bottleneck Analysis
    camera_util_overall = min(100.0, ((total_captured * cam_service_time_mean) / (num_cameras * total_minutes)) * 100.0)
    doctor_util_overall = min(100.0, ((total_doctor_reviewed) / (num_doctors * (doctor_capacity_per_hour / 60.0) * total_minutes)) * 100.0)
    ai_util_overall = min(100.0, ((total_gradable) / ((ai_capacity_per_hour / 60.0) * total_minutes)) * 100.0)

    if doctor_queue_history[-1] > 10:
        bottleneck = "Tele-Ophthalmologist Review Capacity (Queue buildup detected; add reviewing clinician)"
    elif current_camera_queue > 10:
        bottleneck = "Camera Acquisition Capacity (Long intake queue; add portable fundus camera)"
    elif bandwidth_mbps < 0.5:
        bottleneck = "Telemedicine Bandwidth (High upload latency; deploy edge offline AI inference)"
    else:
        bottleneck = "Balanced Rural Screening Flow (No critical bottleneck)"

    results = {
        "parameters": {
            "arrival_rate_per_hr": arrival_rate,
            "num_cameras": num_cameras,
            "image_rejection_rate_pct": image_rejection_rate,
            "bandwidth_mbps": bandwidth_mbps,
            "ai_capacity_per_hr": ai_capacity_per_hour,
            "doctor_capacity_per_hr": doctor_capacity_per_hour,
            "num_doctors": num_doctors,
            "simulation_hours": simulation_hours,
            "referral_rate_pct": referral_rate
        },
        "kpis": {
            "total_patients_arrived": int(total_arrivals),
            "total_images_captured": int(total_captured),
            "images_captured_per_hour": round(total_captured / simulation_hours, 1),
            "image_rejection_rate": f"{image_rejection_rate:.1f}%",
            "ungradable_images_rejected": int(total_rejected),
            "recaptured_successfully": int(total_recaptured),
            "gradable_screened": int(total_gradable),
            "referable_detected": int(total_referable),
            "non_referable_routine": int(total_non_referable),
            "cases_reviewed_by_doctor": int(total_doctor_reviewed),
            "final_doctor_backlog": int(current_doc_queue),
            "average_waiting_time_min": round(float(np.mean(wait_time_history)), 1),
            "peak_doctor_queue": int(max(doctor_queue_history) if doctor_queue_history else 0),
            "camera_utilization_pct": round(camera_util_overall, 1),
            "doctor_utilization_pct": round(doctor_util_overall, 1),
            "ai_utilization_pct": round(ai_util_overall, 1),
            "bottleneck_diagnosis": bottleneck
        },
        "time_series": {
            "time_steps_hours": [round(t / 60.0, 2) for t in time_steps],
            "cumulative_arrivals": cumulative_arrivals_history,
            "camera_queue_length": camera_queue_history,
            "doctor_priority_queue": doctor_queue_history,
            "waiting_time_minutes": wait_time_history,
            "ai_throughput_15min": ai_throughput_history,
            "doctor_utilization_pct": doc_utilization_history
        }
    }

    # Save to disk as default cache
    try:
        SIMULATION_DIR.mkdir(parents=True, exist_ok=True)
        with open(DEFAULT_SIM_OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
    except Exception as e:
        print(f"Warning: Could not save simulation output: {e}")

    return results


def load_simulation_results() -> Dict[str, Any]:
    """Loads cached simulation results or runs baseline simulation if not present."""
    if DEFAULT_SIM_OUTPUT_PATH.exists():
        try:
            with open(DEFAULT_SIM_OUTPUT_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return run_rural_deployment_simulation()
