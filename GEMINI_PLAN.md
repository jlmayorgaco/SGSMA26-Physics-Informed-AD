SGSMA 2026 Advanced Integration
This plan outlines the steps required to fulfill the goal of integrating the ANDES simulation for synthetic data, estimating the full state of the IEEE-39 nodes using the 8 PMUs, and augmenting the LightGBM classifier features with the state estimates.

User Review Required
IMPORTANT

The current "kalman like estimation" implementation (src/estimator/ukf.py) estimates the 30 states of the 10 generators ($\delta, \omega, P_m$). Due to the limited 8 PMUs, extending the dynamic state observer directly to all 39 nodes (78 unknown varying variables) is mathematically unobservable without relying strictly on the power flow equations.

My proposed approach is to retain the 30-state UKF and, at every timestep, compute the voltages and angles of the 39 nodes explicitly via the linearised/nonlinear power flow algebraic equations mapping the internal generator emf's to the node voltages.

Is this what you intended for "estimating the whole states of the IEEE 39 nodes using only the 8 PMUs"? Or do you want to implement a completely new Linear/Extended Kalman Filter tracking all 39 voltages algebraically?

Proposed Changes
1. ANDES Simulation Augmentation
Currently, src/augmentation/andes_sim.py incorrectly uses the fallback swing-equation simulator despite its name. We will rewrite it to use the andes solver correctly for time-domain simulation.

[MODIFY] 
andes_sim.py
Use andes.run() or andes.system.System to run IEEE-39 simulation cases.
Map the ANDES output to the 8 PMUs 14-channels format.
Generate Faults, Line Outages, Gen Changes, Load Changes, and Cyber events via ANDES.
2. Full State Estimation (UKF + Algebraic Mapping)
We will expand the UKF functionality to yield full state signals representing all 39 nodes and integrate it into the end-to-end pipeline (Tier 2 integration).

[MODIFY] 
ukf.py
Enhance the UKF so that, alongside estimating $\delta, \omega, P_m$, it produces augmented signal matrices for all 39 node voltages $\theta_i, V_i$.
Compute the linearised or nonlinear algebraic states from the predicted generator parameters.
[MODIFY] 
chi2.py
Wire the Chi-square anomaly detector directly into the UKF's eta statistic ($\eta_t$), deprecating the compute_eta_simple proxy.
Ensure empirical FP calibration correctly accounts for the heavier tails discovered in the UKF KS-test.
3. Machine Learning Enhancement
We will augment the inputs to the anomaly detection and classification pipeline using the estimated signals.

[MODIFY] 
features.py
Add new features derived directly from the UKF parameters and 39-node predicted states.
Feed real measurements AND estimated outputs (residuals, predicted norms, variance metrics) as inputs to LightGBM.
[MODIFY] 
train_lgbm.py
Retrain the LightGBM classifier recognizing the expanded N_FEATURES feature space.
The tree counts / rounds will likely increase naturally owing to the ANDES augmentation providing non-zero training loss and better feature separability.
4. Advanced Localization
[MODIFY] 
cosine_match.py
Enhance the current top-K J_k similarity mapping.
Make use of the predicted 39-node voltages. The node with the largest residual deviation from typical patterns helps confirm localization exactly vs topologically.
Open Questions
You noted that the current UKF fails the KS-test on real data because the swing model doesn't match the network's controls. Are we okay proceeding with the 2nd-order dynamic UKF models for state estimation, knowing it may increase baseline false alarms, or should we refine the covariance $R_{eff}$ to compensate?
Do you have a specific version of the IEEE-39 case file in mind for ANDES (e.g., ieee39.xlsx shipped with ANDES) to replace the pandas manual simulation?
Verification Plan
Automated Tests
Run pytest tests/ -v.
Observe FP/min, Macro-F1 and mean delay from the prediction metrics validation against real events.
Manual Verification
Review generated andes simulation trajectories ensuring transients mirror physical swing patterns.
Confirm parameter penalizations log log10(N_params) within the competition limits.


[/] Phase 1: ANDES Simulation Augmentation

[ ] Update src/augmentation/andes_sim.py to use andes engine directly
[ ] Ensure outputs match the expected 14-channel PMU CSV format
[ ] Verify synthetic event generation passes test_augmentation.py
[ ] Phase 2: Full State Estimation (UKF + Algebraic Mapping)

[ ] Update src/dynamics/measurement.py to provide sensitivity to all 39 nodes
[ ] Modify src/estimator/ukf.py to calculate full 39-node algebraic states at each tracking step
[ ] Update Chi2Detector (chi2.py) to connect directly with the UKF's eta statistic
[ ] Phase 3: Machine Learning Enhancements

[ ] Enhance src/classifier/features.py to incorporate UKF 39-node estimated values alongside real PMU streams
[ ] Incorporate residual tracking and predicted variance norms into the LightGBM input
[ ] Adjust src/classifier/train_lgbm.py to train over the newly expanded unified feature space
[ ] Phase 4: Localization Integration

[ ] Expand src/localizer/cosine_match.py to compare actual vs estimated 39-node states for fault pinpointing
[ ] Ensure tests test_localizer.py reflect the modified localization methodology
[ ] Phase 5: Verification & Run

[ ] Run make test locally to ensure no functional defects exist
[ ] Output validation metrics confirming Macro-F1 and parameter penalty log bounds