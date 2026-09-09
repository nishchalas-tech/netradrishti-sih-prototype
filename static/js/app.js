/**
 * NETRADRISHTI — Frontend Application Controller
 * Smart India Hackathon 2026 | Problem Statement: SIH26038
 * Formal Clinical Light Workstation Logic
 */

// Application Global State
const state = {
    activeTab: 'RURAL CLINIC',
    currentImageData: null,
    currentCaseId: null,
    selectedDoctorCaseId: null,
    queueFilter: 'all',
    simulationData: null,
    chartArrivals: null,
    chartQueue: null
};

// Initialize Application on Page Load
document.addEventListener('DOMContentLoaded', () => {
    checkSystemStatus();
    setupDropZone();
    loadDoctorQueue();
    initDefaultSimulation();
});

// -------------------------------------------------------------
// 1. Navigation & System Status
// -------------------------------------------------------------
function switchTab(tabName) {
    state.activeTab = tabName;

    // Update Nav Tabs
    document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.screen-view').forEach(s => s.classList.remove('active'));

    if (tabName === 'RURAL CLINIC') {
        document.getElementById('tab-rural').classList.add('active');
        document.getElementById('screen-rural').classList.add('active');
    } else if (tabName === 'DOCTOR REVIEW') {
        document.getElementById('tab-doctor').classList.add('active');
        document.getElementById('screen-doctor').classList.add('active');
        loadDoctorQueue();
    } else if (tabName === 'SIMULATION') {
        document.getElementById('tab-sim').classList.add('active');
        document.getElementById('screen-sim').classList.add('active');
        if (!state.simulationData) {
            initDefaultSimulation();
        }
    }
}

async function checkSystemStatus() {
    const indicator = document.getElementById('system-status-indicator');
    const statusText = document.getElementById('system-status-text');

    try {
        const res = await fetch('/api/status');
        const data = await res.json();

        if (data.model_loaded) {
            indicator.style.color = '#059669';
            indicator.style.borderColor = 'rgba(5, 150, 105, 0.4)';
            indicator.style.background = 'rgba(16, 185, 129, 0.15)';
            statusText.textContent = 'AI PIPELINE ONLINE (NetraNet)';
        } else {
            indicator.style.color = '#dc2626';
            indicator.style.borderColor = 'rgba(220, 38, 38, 0.4)';
            indicator.style.background = 'rgba(239, 68, 68, 0.15)';
            statusText.textContent = 'MODEL NOT LOADED';
        }
    } catch (e) {
        statusText.textContent = 'OFFLINE / SERVER ERROR';
    }
}

// -------------------------------------------------------------
// 2. Image Acquisition & Drag-and-Drop
// -------------------------------------------------------------
function setupDropZone() {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropZone.style.borderColor = '#2563eb';
            dropZone.style.backgroundColor = '#eff6ff';
        });
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropZone.style.borderColor = '#cbd5e1';
            dropZone.style.backgroundColor = '#f8fafc';
        });
    });

    dropZone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files && files.length > 0) {
            handleUploadedFile(files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (fileInput.files && fileInput.files.length > 0) {
            handleUploadedFile(fileInput.files[0]);
        }
    });
}

function handleUploadedFile(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
        setCurrentImage(e.target.result);
        document.getElementById('demo-indicator').style.display = 'none';
    };
    reader.readAsDataURL(file);
}

function setCurrentImage(dataUri) {
    state.currentImageData = dataUri;
    const preview = document.getElementById('fundus-preview');
    const placeholder = document.getElementById('preview-placeholder');
    const btnCheck = document.getElementById('btn-check-quality');

    preview.src = dataUri;
    preview.style.display = 'block';
    placeholder.style.display = 'none';
    btnCheck.disabled = false;

    // Reset downstream quality and AI blocks
    resetQualityResults();
}

function resetQualityResults() {
    document.getElementById('quality-gate-idle').style.display = 'block';
    document.getElementById('quality-ungradable').style.display = 'none';
    document.getElementById('quality-accepted').style.display = 'none';
    document.getElementById('quality-metrics-grid').style.display = 'none';
    document.getElementById('ai-action-container').style.display = 'none';
    document.getElementById('ai-results-card').style.display = 'none';
}

function resetForm() {
    state.currentImageData = null;
    state.currentCaseId = null;

    document.getElementById('pat-id').value = `PAT-RUR-${Math.floor(1000 + Math.random() * 9000)}`;
    document.getElementById('pat-age').value = 54;
    document.getElementById('pat-notes').value = 'Routine diabetic screening checkup.';

    document.getElementById('fundus-preview').style.display = 'none';
    document.getElementById('fundus-preview').src = '';
    document.getElementById('preview-placeholder').style.display = 'block';
    document.getElementById('btn-check-quality').disabled = true;
    document.getElementById('demo-indicator').style.display = 'none';

    resetQualityResults();
}

// -------------------------------------------------------------
// 3. Hackathon 60-Second Demo Case Loader
// -------------------------------------------------------------
async function loadDemoCase(caseId) {
    try {
        const res = await fetch(`/api/demo-case/${caseId}`);
        if (!res.ok) throw new Error('Could not load demo case');
        const data = await res.json();

        // Populate Form
        document.getElementById('pat-id').value = data.patient_id;
        document.getElementById('pat-age').value = data.age;
        document.getElementById('pat-notes').value = data.clinical_notes;

        // Display Demo Badge
        const demoInd = document.getElementById('demo-indicator');
        demoInd.style.display = 'block';
        document.getElementById('demo-case-title').textContent = data.name;

        // Set Image
        setCurrentImage(data.image_data);

        // Switch to Rural Clinic View
        switchTab('RURAL CLINIC');
    } catch (e) {
        alert('Error loading demo case: ' + e.message);
    }
}

// -------------------------------------------------------------
// 4. Automated Image Quality Gate
// -------------------------------------------------------------
async function runQualityCheck() {
    if (!state.currentImageData) return;

    const btn = document.getElementById('btn-check-quality');
    btn.disabled = true;
    btn.textContent = '⏳ EVALUATING GRADABILITY...';

    try {
        const res = await fetch('/api/quality-check', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image_data: state.currentImageData })
        });

        if (!res.ok) throw new Error('Quality assessment failed');
        const data = await res.json();

        document.getElementById('quality-gate-idle').style.display = 'none';
        document.getElementById('quality-metrics-grid').style.display = 'grid';
        document.getElementById('kpi-quality-score').textContent = `${data.quality_score}/100`;
        document.getElementById('kpi-blur-score').textContent = `${data.blur_score}`;
        document.getElementById('kpi-illum-score').textContent = `${data.illumination_score}`;

        if (!data.usable) {
            // UNGRADABLE
            document.getElementById('quality-ungradable').style.display = 'block';
            document.getElementById('quality-accepted').style.display = 'none';
            document.getElementById('ungradable-reason-text').textContent = data.reason || 'Image too blurry';
            document.getElementById('ai-action-container').style.display = 'none';
        } else {
            // ACCEPTED
            document.getElementById('quality-accepted').style.display = 'block';
            document.getElementById('quality-ungradable').style.display = 'none';
            document.getElementById('accepted-desc-text').textContent = data.recommendation;
            document.getElementById('ai-action-container').style.display = 'block';
        }
    } catch (e) {
        alert('Quality Check Error: ' + e.message);
    } finally {
        btn.disabled = false;
        btn.textContent = '🔍 CHECK IMAGE QUALITY';
    }
}

// -------------------------------------------------------------
// 5. AI Screening & Grad-CAM Explainability
// -------------------------------------------------------------
async function runAiScreening() {
    if (!state.currentImageData) return;

    const btn = document.getElementById('btn-run-ai');
    btn.disabled = true;
    btn.textContent = '🧠 COMPUTING NETRANET INFERENCE & GRAD-CAM...';

    const payload = {
        image_data: state.currentImageData,
        patient_id: document.getElementById('pat-id').value,
        age: parseInt(document.getElementById('pat-age').value) || 50,
        clinical_notes: document.getElementById('pat-notes').value
    };

    try {
        const res = await fetch('/api/run-screening', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (res.status === 503) {
            const err = await res.json();
            alert(`⛔ ${err.detail}`);
            return;
        }

        if (!res.ok) {
            const err = await res.json();
            alert(`Error: ${JSON.stringify(err.detail || err)}`);
            return;
        }

        const data = await res.json();
        state.currentCaseId = data.case_id;

        // Render AI Screening Card
        const aiCard = document.getElementById('ai-results-card');
        aiCard.style.display = 'block';
        document.getElementById('case-id-badge').textContent = data.case_id;

        const banner = document.getElementById('triage-banner');
        const predText = document.getElementById('triage-prediction-text');
        const subText = document.getElementById('triage-subtitle');
        const prioVal = document.getElementById('triage-priority-val');

        if (data.prediction === 'REFERABLE') {
            banner.className = 'triage-box triage-referable';
            predText.textContent = 'REFERABLE DR';
            subText.textContent = 'High Risk of Diabetic Retinopathy Pathology Detected';
            prioVal.textContent = 'HIGH PRIORITY';
            prioVal.style.color = '#dc2626';
        } else {
            banner.className = 'triage-box triage-nonreferable';
            predText.textContent = 'NON-REFERABLE';
            subText.textContent = 'Low Risk / Routine 12-Month Follow-Up Recommended';
            prioVal.textContent = 'ROUTINE';
            prioVal.style.color = '#2563eb';
        }

        document.getElementById('model-conf-val').textContent = `${(data.confidence * 100).toFixed(1)}%`;

        // Render 3-Way Explainability Images
        document.getElementById('xai-img-orig').src = data.images.original;
        document.getElementById('xai-img-heatmap').src = data.images.heatmap;
        document.getElementById('xai-img-overlay').src = data.images.overlay;

        document.getElementById('evidence-text-desc').innerHTML = `
            <strong>Attribution Evidence:</strong> ${data.evidence_text}
        `;

        // Update Doctor Queue in background
        loadDoctorQueue();

    } catch (e) {
        alert('AI Screening Error: ' + e.message);
    } finally {
        btn.disabled = false;
        btn.textContent = '🚀 RUN AI SCREENING';
    }
}

// -------------------------------------------------------------
// 6. Doctor Review Dashboard & Queues
// -------------------------------------------------------------
async function loadDoctorQueue(filter = null) {
    if (filter) state.queueFilter = filter;

    try {
        const res = await fetch('/api/cases');
        if (!res.ok) return;
        const data = await res.json();

        // Update Counters
        document.getElementById('stat-high-count').textContent = data.high_priority_count;
        document.getElementById('stat-routine-count').textContent = data.routine_count;
        document.getElementById('stat-total-count').textContent = data.total_count;

        const reviewed = data.all_cases.filter(c => c.doctor_review.status !== 'PENDING REVIEW').length;
        document.getElementById('stat-reviewed-count').textContent = reviewed;
        document.getElementById('queue-badge-counter').textContent = data.high_priority_count;

        // Filter Table Rows
        let displayList = data.all_cases;
        if (state.queueFilter === 'HIGH PRIORITY') {
            displayList = data.high_priority_queue;
        } else if (state.queueFilter === 'ROUTINE') {
            displayList = data.routine_queue;
        }

        const tbody = document.getElementById('queue-table-body');
        if (displayList.length === 0) {
            tbody.innerHTML = '<tr><td colspan="5" style="text-align:center; color:#64748b;">No cases in this queue.</td></tr>';
            return;
        }

        tbody.innerHTML = displayList.map(c => {
            const isHigh = c.priority === 'HIGH PRIORITY';
            const badgeClass = isHigh ? 'badge-priority-high' : 'badge-priority-routine';
            const isSelected = c.case_id === state.selectedDoctorCaseId;

            return `
                <tr class="${isSelected ? 'selected' : ''}" onclick="selectCaseForReview('${c.case_id}')">
                    <td style="font-family:monospace; font-weight:700;">${c.case_id}</td>
                    <td>${c.patient_id} (${c.age}y)</td>
                    <td><span class="${badgeClass}">${c.priority}</span></td>
                    <td style="font-weight:600;">${c.ai_result ? c.ai_result.prediction : 'Pending'}</td>
                    <td>${formatDecisionBadge(c.doctor_review.status)}</td>
                </tr>
            `;
        }).join('');

        // If no case currently selected, select the first high priority or first case
        if (!state.selectedDoctorCaseId && displayList.length > 0) {
            selectCaseForReview(displayList[0].case_id);
        }

    } catch (e) {
        console.error('Failed to load queue:', e);
    }
}

function filterQueue(f) {
    loadDoctorQueue(f);
}

function formatDecisionBadge(status) {
    if (status === 'CONFIRMED REFERRAL') {
        return '<span style="background:#ecfdf5; color:#065f46; border:1px solid #a7f3d0; padding:2px 6px; border-radius:4px; font-weight:700; font-size:0.75rem;">CONFIRMED</span>';
    } else if (status === 'ROUTINE FOLLOW-UP') {
        return '<span style="background:#eff6ff; color:#1e3a8a; border:1px solid #bfdbfe; padding:2px 6px; border-radius:4px; font-weight:700; font-size:0.75rem;">FOLLOW-UP</span>';
    } else if (status === 'REQUEST RECERTIFICATION') {
        return '<span style="background:#fef3c7; color:#92400e; border:1px solid #fde68a; padding:2px 6px; border-radius:4px; font-weight:700; font-size:0.75rem;">RECERT</span>';
    }
    return '<span style="background:#f1f5f9; color:#64748b; padding:2px 6px; border-radius:4px; font-size:0.75rem;">PENDING</span>';
}

async function selectCaseForReview(caseId) {
    state.selectedDoctorCaseId = caseId;
    document.getElementById('doc-case-badge').textContent = caseId;

    try {
        const res = await fetch(`/api/case/${caseId}`);
        if (!res.ok) throw new Error('Case not found');
        const caseData = await res.json();

        document.getElementById('doc-empty-state').style.display = 'none';
        document.getElementById('doc-review-content').style.display = 'block';

        // Demographics
        document.getElementById('doc-pat-id').textContent = caseData.patient_id;
        document.getElementById('doc-pat-age').textContent = caseData.age;
        document.getElementById('doc-reg-time').textContent = caseData.created_at;
        document.getElementById('doc-pat-notes').textContent = caseData.clinical_notes;

        // Imaging
        document.getElementById('doc-fundus-img').src = caseData.image_data || '';
        document.getElementById('doc-overlay-img').src = caseData.overlay_data || caseData.image_data || '';

        // AI Info
        document.getElementById('doc-ai-pred').textContent = caseData.ai_result ? caseData.ai_result.prediction : '--';
        document.getElementById('doc-ai-conf').textContent = caseData.ai_result ? `${(caseData.ai_result.confidence * 100).toFixed(1)}%` : '--';
        document.getElementById('doc-quality-score').textContent = caseData.quality ? caseData.quality.quality_score : '--';

        // Decision section
        const statusBadge = document.getElementById('decision-status-badge');
        statusBadge.textContent = caseData.doctor_review.status;
        if (caseData.doctor_review.status === 'CONFIRMED REFERRAL') {
            statusBadge.style.background = '#ecfdf5';
            statusBadge.style.color = '#065f46';
        } else if (caseData.doctor_review.status === 'ROUTINE FOLLOW-UP') {
            statusBadge.style.background = '#eff6ff';
            statusBadge.style.color = '#1e3a8a';
        } else {
            statusBadge.style.background = '#e2e8f0';
            statusBadge.style.color = '#334155';
        }

        document.getElementById('doc-notes-input').value = caseData.doctor_review.notes || '';
        
        if (caseData.doctor_review.timestamp) {
            document.getElementById('decision-audit-trail').textContent = 
                `Recorded by ${caseData.doctor_review.doctor_id} at ${caseData.doctor_review.timestamp}`;
        } else {
            document.getElementById('decision-audit-trail').textContent = 'Decision pending review.';
        }

        // Highlight selected row in table
        loadDoctorQueue();

    } catch (e) {
        console.error('Error loading case detail:', e);
    }
}

async function submitDoctorDecision(decision) {
    if (!state.selectedDoctorCaseId) return;

    const notes = document.getElementById('doc-notes-input').value;

    try {
        const res = await fetch('/api/doctor-decision', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                case_id: state.selectedDoctorCaseId,
                decision: decision,
                notes: notes,
                doctor_id: 'Dr. S. Sharma (AIIMS-DL Tele-Ophthalmology)'
            })
        });

        if (!res.ok) throw new Error('Failed to record clinical decision');
        const data = await res.json();

        alert(`Clinical Decision Recorded: ${decision}`);
        selectCaseForReview(state.selectedDoctorCaseId);
        loadDoctorQueue();
    } catch (e) {
        alert('Error saving decision: ' + e.message);
    }
}

// -------------------------------------------------------------
// 7. System / Simulink Simulation Dashboard
// -------------------------------------------------------------
function updateSliderVal(name) {
    const val = document.getElementById(`slider-${name}`).value;
    document.getElementById(`val-${name}`).textContent = val;
}

async function initDefaultSimulation() {
    try {
        const res = await fetch('/api/simulation/default');
        if (!res.ok) return;
        const data = await res.json();
        renderSimulationTelemetry(data);
    } catch (e) {
        console.error('Failed to load default simulation:', e);
    }
}

async function executeSimulation() {
    const payload = {
        arrival_rate: parseFloat(document.getElementById('slider-arrival').value),
        num_cameras: parseInt(document.getElementById('slider-cameras').value),
        bandwidth_mbps: parseFloat(document.getElementById('slider-bandwidth').value),
        ai_capacity_per_hour: parseFloat(document.getElementById('slider-ai-cap').value),
        num_doctors: parseInt(document.getElementById('slider-doctors').value),
        doctor_capacity_per_hour: parseFloat(document.getElementById('slider-doc-cap').value),
        image_rejection_rate: parseFloat(document.getElementById('slider-rejection').value),
        simulation_hours: parseFloat(document.getElementById('slider-duration').value)
    };

    try {
        const res = await fetch('/api/run-simulation', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (!res.ok) throw new Error('Simulation failed');
        const data = await res.json();
        renderSimulationTelemetry(data);
    } catch (e) {
        alert('Simulation Error: ' + e.message);
    }
}

function renderSimulationTelemetry(data) {
    state.simulationData = data;
    const kpis = data.kpis;

    // Render KPI Cards
    document.getElementById('sim-kpi-arrivals').textContent = kpis.total_patients_arrived;
    document.getElementById('sim-kpi-hourly-capture').textContent = kpis.images_captured_per_hour;
    document.getElementById('sim-kpi-rejection').textContent = kpis.image_rejection_rate;
    document.getElementById('sim-kpi-doc-done').textContent = kpis.cases_reviewed_by_doctor;
    document.getElementById('sim-kpi-wait').textContent = `${kpis.average_waiting_time_min} m`;
    document.getElementById('sim-kpi-referrals').textContent = kpis.referable_detected;
    document.getElementById('sim-kpi-recaptured').textContent = kpis.recaptured_successfully;
    document.getElementById('sim-kpi-backlog').textContent = kpis.final_doctor_backlog;

    // Bottleneck Alert
    const bAlert = document.getElementById('sim-bottleneck-alert');
    bAlert.innerHTML = `<strong>Operational Bottleneck Diagnosis:</strong> ${kpis.bottleneck_diagnosis}`;
    if (kpis.bottleneck_diagnosis.toLowerCase().includes('bottleneck')) {
        bAlert.style.borderLeftColor = '#dc2626';
        bAlert.style.background = '#fef2f2';
    } else {
        bAlert.style.borderLeftColor = '#059669';
        bAlert.style.background = '#ecfdf5';
    }

    // Resource Utilization Progress Bars
    document.getElementById('lbl-util-cam').textContent = `${kpis.camera_utilization_pct}%`;
    document.getElementById('bar-util-cam').style.width = `${kpis.camera_utilization_pct}%`;

    document.getElementById('lbl-util-ai').textContent = `${kpis.ai_utilization_pct}%`;
    document.getElementById('bar-util-ai').style.width = `${kpis.ai_utilization_pct}%`;

    document.getElementById('lbl-util-doc').textContent = `${kpis.doctor_utilization_pct}%`;
    document.getElementById('bar-util-doc').style.width = `${kpis.doctor_utilization_pct}%`;

    // Render Charts
    renderCharts(data.time_series);
}

function renderCharts(ts) {
    const timeLabels = ts.time_steps_hours.map(h => `${h}h`);

    // Chart 1: Arrivals vs Camera Queue
    const ctx1 = document.getElementById('chart-arrivals').getContext('2d');
    if (state.chartArrivals) state.chartArrivals.destroy();

    state.chartArrivals = new Chart(ctx1, {
        type: 'line',
        data: {
            labels: timeLabels,
            datasets: [
                {
                    label: 'Cumulative Patient Arrivals',
                    data: ts.cumulative_arrivals,
                    borderColor: '#2563eb',
                    backgroundColor: 'rgba(37, 99, 235, 0.08)',
                    fill: true,
                    tension: 0.2
                },
                {
                    label: 'Camera Queue Length',
                    data: ts.camera_queue_length,
                    borderColor: '#d97706',
                    backgroundColor: 'transparent',
                    borderDash: [5, 5],
                    tension: 0.2
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { position: 'top' } },
            scales: {
                x: { title: { display: true, text: 'Clinical Shift Timeline (Hours)' } },
                y: { title: { display: true, text: 'Patients' } }
            }
        }
    });

    // Chart 2: Doctor Priority Queue vs Waiting Time
    const ctx2 = document.getElementById('chart-queue').getContext('2d');
    if (state.chartQueue) state.chartQueue.destroy();

    state.chartQueue = new Chart(ctx2, {
        type: 'line',
        data: {
            labels: timeLabels,
            datasets: [
                {
                    label: 'Doctor Priority Queue (Patients)',
                    data: ts.doctor_priority_queue,
                    borderColor: '#dc2626',
                    backgroundColor: 'rgba(220, 38, 38, 0.08)',
                    fill: true,
                    yAxisID: 'y'
                },
                {
                    label: 'Avg Patient Wait Time (Minutes)',
                    data: ts.waiting_time_minutes,
                    borderColor: '#059669',
                    backgroundColor: 'transparent',
                    yAxisID: 'y1'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { position: 'top' } },
            scales: {
                x: { title: { display: true, text: 'Clinical Shift Timeline (Hours)' } },
                y: { type: 'linear', position: 'left', title: { display: true, text: 'Doctor Queue (Patients)' } },
                y1: { type: 'linear', position: 'right', grid: { drawOnChartArea: false }, title: { display: true, text: 'Wait Time (Min)' } }
            }
        }
    });
}

function exportSimulationJson() {
    if (!state.simulationData) return;
    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(state.simulationData, null, 2));
    const downloadAnchor = document.createElement('a');
    downloadAnchor.setAttribute("href", dataStr);
    downloadAnchor.setAttribute("download", "netradrishti_simulink_simulation_output.json");
    document.body.appendChild(downloadAnchor);
    downloadAnchor.click();
    downloadAnchor.remove();
}
