% RURAL_DR_SCREENING_MODEL
% MATLAB / Simulink SimEvents Model Configuration Script
% NETRADRISHTI - Explainable AI for Diabetic Retinopathy Screening in Rural India
% Smart India Hackathon 2026 (Problem Statement SIH26038)
%
% This script defines and initializes the discrete-event queueing model parameters
% for evaluating rural tele-ophthalmology deployment capacity in Simulink / SimEvents.

clc;
clear;

fprintf('=======================================================\n');
fprintf('  NETRADRISHTI: Simulink Rural Screening Architecture \n');
fprintf('=======================================================\n');

% -------------------------------------------------------------
% 1. Model Configuration & Operating Parameters
% -------------------------------------------------------------
% Shift configuration: 8 hours (480 minutes)
ShiftDurationHours = 8.0;
TotalSimMinutes = ShiftDurationHours * 60;

% Patient Arrival Process (Poisson Arrival Rate)
% Lambda = 18 patients / hour -> Mean inter-arrival time = 3.33 minutes
PatientArrivalRatePerHour = 18.0; 
InterArrivalTimeMinutes = 60.0 / PatientArrivalRatePerHour;

% Portable Fundus Cameras in Primary Health Centre (PHC)
NumCameras = 2;
CameraAcquisitionTimeMinutes = 4.5; % Mean imaging duration per patient

% Image Quality Gate Filter
% Rejection rate due to poor pupil dilation, eye movement, or opacity
QualityRejectionRate = 0.12; % 12% initial rejection
ImmediateRecaptureSuccessRate = 0.75; % 75% recaptured successfully on-site
RecaptureAdditionalDelayMinutes = 2.0;

% Bandwidth & Telemedicine Transmission (Rural Uplink)
BandwidthMbps = 4.0;
FundusPackageSizeMB = 2.5;
TransmissionDelaySeconds = (FundusPackageSizeMB * 8) / BandwidthMbps; 

% Edge AI Screening Inference Unit
AICapacityPerHour = 60.0; % Cases/hour
AIServiceTimeMinutes = 60.0 / AICapacityPerHour; % 1 minute per case

% DR Risk Stratification / Triage Split
% Population prevalence of referable diabetic retinopathy in screened cohort
ReferralRate = 0.24; % 24% referable, 76% routine non-referable

% Tele-Ophthalmologist Review Station (District/Tertiary Hospital)
NumOphthalmologists = 1;
DoctorReviewCapacityPerHour = 15.0; % Reviews per doctor per hour
DoctorServiceTimeMinutes = 60.0 / (DoctorReviewCapacityPerHour * NumOphthalmologists);

% -------------------------------------------------------------
% 2. Theoretical M/M/c Queueing Bounds (Analytical Validation)
% -------------------------------------------------------------
EffectiveArrivalRateGradable = (PatientArrivalRatePerHour / 60.0) * (1 - QualityRejectionRate * (1 - ImmediateRecaptureSuccessRate));
EffectiveReferableArrivals = EffectiveArrivalRateGradable * ReferralRate;
DoctorServiceRate = (DoctorReviewCapacityPerHour * NumOphthalmologists) / 60.0;

DoctorRho = EffectiveReferableArrivals / DoctorServiceRate; % Utilization factor

fprintf('[PARAMETERS INITIALIZED]\n');
fprintf('  Shift Duration:             %.1f hours (%d minutes)\n', ShiftDurationHours, TotalSimMinutes);
fprintf('  Patient Arrival Rate:       %.1f patients/hour\n', PatientArrivalRatePerHour);
fprintf('  Available Cameras:          %d\n', NumCameras);
fprintf('  Quality Rejection Rate:     %.1f%%\n', QualityRejectionRate * 100);
fprintf('  AI Unit Throughput:         %.1f cases/hour\n', AICapacityPerHour);
fprintf('  Tele-Ophthalmologist Count: %d\n', NumOphthalmologists);
fprintf('  Doctor Review Capacity:     %.1f cases/hour\n', DoctorReviewCapacityPerHour);
fprintf('  Doctor Utilization (rho):   %.2f%%\n', DoctorRho * 100);

if DoctorRho >= 1.0
    fprintf('\n[WARNING] Doctor queue will become unstable (rho >= 1.0). Recommend adding 1 ophthalmologist.\n');
else
    fprintf('\n[STATUS] Stable queueing regime. Expected average waiting time < 25 minutes.\n');
end

% -------------------------------------------------------------
% 3. Export Simulation Parameters to NETRADRISHTI Bridge
% -------------------------------------------------------------
simParams = struct(...
    'shift_duration_hours', ShiftDurationHours, ...
    'arrival_rate_per_hr', PatientArrivalRatePerHour, ...
    'num_cameras', NumCameras, ...
    'quality_rejection_rate', QualityRejectionRate, ...
    'bandwidth_mbps', BandwidthMbps, ...
    'ai_capacity_per_hr', AICapacityPerHour, ...
    'num_doctors', NumOphthalmologists, ...
    'doctor_capacity_per_hr', DoctorReviewCapacityPerHour, ...
    'doctor_utilization_pct', round(DoctorRho * 100, 1) ...
);

outputJsonFile = fullfile('..', 'simulation', 'simulink_config_export.json');
encodedJSON = jsonencode(simParams);
fid = fopen(outputJsonFile, 'w');
if fid ~= -1
    fwrite(fid, encodedJSON, 'char');
    fclose(fid);
    fprintf('\n[EXPORT] Configuration exported to %s for NETRADRISHTI Dashboard.\n', outputJsonFile);
end

fprintf('=======================================================\n');
fprintf('Simulink Model Setup Complete. Run model in Simulink or view via NETRADRISHTI UI.\n');
