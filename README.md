# AI-Powered Satellite Attitude Determination and Control System (Hybrid ADCS)

## Overview
This project implements a Hybrid AI + Kalman Filter-based Attitude Determination and Control System (ADCS) for small satellites. The goal is to maintain precise satellite orientation despite external disturbances and limited onboard computation, achieving research-grade spacecraft control performance.

## Problem Statement
Small satellites (CubeSats) experience attitude drift due to:
- Environmental disturbances (gravity gradient, drag, solar pressure)
- Actuator limitations
- Sensor noise and limited computational power

Traditional PD/LQR controllers struggle with unmodeled disturbances and nonlinear dynamics. This project develops a hybrid controller combining:
- Kalman Filtering (EKF) for state estimation  
- AI-based Neural Compensator for disturbance rejection and adaptive control  

## Objectives
- Design a 3-axis satellite attitude simulation using quaternion dynamics  
- Implement an Extended Kalman Filter (EKF) for state estimation  
- Develop a Neural Network-based control augmentation on top of PD control  
- Validate the Hybrid ADCS under realistic noise and disturbances  
- Compare with classical PD/LQR control performance  

## System Architecture
`
+-----------------------------+
|     Star Tracker / Gyro     |
+---------------+-------------+
                |
                v
      +---------+---------+
      |   Kalman Filter   |
      |        (EKF)      |
      +---------+---------+
                |
      Estimated Attitude & Rates
                |
                v
   +------------+------------+
   | Hybrid Controller (PD + AI) |
   +------------+------------+
                |
                v
      +---------+---------+
      | Satellite Dynamics |
      +---------+---------+
                |
                v
  +-------------+-------------+
  | Reaction Wheel Actuators  |
  +---------------------------+
`

## Control Strategy
- **Baseline:** PD Controller for nominal attitude control.  
- **AI Augmentation:** Neural network compensator trained to minimize residual attitude error caused by unmodeled dynamics and disturbances.  
- **Estimator:** Quaternion-based EKF provides accurate estimates of attitude and angular velocity.  

### Hybrid Control Law
\[
\tau_{hybrid} = \tau_{PD} + f_{AI}(q_{err}, \omega)
\]

Where:  
- \( \tau_{PD} \): torque from proportional-derivative control  
- \( f_{AI}(q_{err}, \omega) \): adaptive correction from neural compensator  
- \( q_{err} \): attitude quaternion error  
- \( \omega \): angular velocity vector  

## Simulation Environment
- **Language:** Python  
- **Libraries:** NumPy, SciPy, PyTorch, Matplotlib  
- **Dynamics:** 3-DOF rotational motion with reaction wheel model  
- **Disturbances:** Gravity gradient, drag, and random torque noise  
- **Visualization:** Quaternion and Euler angle evolution, torque profiles, AI-vs-PD comparison plots  

## Results Summary
![Accuracy](https://img.shields.io/badge/RMS_Error-69%25_Improved-success) ![Stability](https://img.shields.io/badge/System-Stable-brightgreen) ![Simulation](https://img.shields.io/badge/Simulation-Python-blue) ![Controller](https://img.shields.io/badge/Controller-Hybrid_AI-orange)

The hybrid AI controller significantly improves pointing accuracy and stability compared to the baseline PD control system.

### Quantitative Metrics
| Metric | PD Controller | Hybrid AI Controller | Improvement |
|--------|----------------|----------------------|-------------|
| RMS Attitude Error | 12.583° | **3.891°** | **69.1%** |
| Max Error | 19.75° | 11.40° | 42% |
| Avg Torque | 0.00457 Nm | 0.00441 Nm | Energy Efficient |
| Stability | Stable | Stable |
| Disturbance Recovery | Moderate | Fast & Adaptive |

### Visual Comparison
#### Attitude Error vs Time
Shows how the hybrid AI controller achieves faster stabilization and smaller steady-state errors compared to the PD controller.  
![Attitude Error Plot](https://github.com/amaranenivinitha/hybrid_adcs/blob/main/results/attitude_error_plot.png?raw=true)

#### Control Torque Profile
Demonstrates that actuator commands remain within torque limits and are smoother in the hybrid control setup.  
![Torque Profile](https://github.com/amaranenivinitha/hybrid_adcs/blob/main/results/torque_profile.png?raw=true)

#### Controller Performance Comparison
The AI-augmented control reduces transient oscillations and improves convergence speed.  
![Controller Comparison](https://github.com/amaranenivinitha/hybrid_adcs/blob/main/results/controller_comparison.png?raw=true)

## Validation Metrics
- **Pointing Accuracy:** RMS error < 0.5° (target)  
- **Estimation Accuracy:** EKF error < 3°  
- **Actuator Efficiency:** No saturation, smooth torque commands  
- **Disturbance Rejection:** Quick recovery after impulse disturbance  
- **Robustness:** Handles sensor noise, dropouts, and torque bias  

## Repository Structure
`
hybrid_adcs/
│
├── src/
│   ├── dynamics.py          # Satellite rotational dynamics
│   ├── ekf.py               # Extended Kalman Filter
│   ├── ai_controller.py     # AI-based neural compensator
│   ├── train_ai.py          # Neural network training script
│   ├── sim.py               # Core simulation script
│   ├── validate_ai.py       # Compare Hybrid AI vs PD controller
│   ├── scale_search.py      # AI torque scale optimization
│   └── utils/               # Quaternion math & helper functions
│
├── results/
│   ├── attitude_error_plot.png
│   ├── torque_profile.png
│   ├── controller_comparison.png
│   ├── metrics_summary.csv
│   └── ai_model.pt
│
├── requirements.txt
├── README.md
└── .gitignore
`

## Future Work
- Implement Unscented Kalman Filter (UKF)  
- Extend to reaction wheels + magnetorquers  
- Integrate Reinforcement Learning (PPO/SAC)  
- Deploy on CubeSat hardware (ARM Cortex-M)  
- Prepare for publication in *Acta Astronautica* or *AIAA GNC*  

## Keywords
ADCS, CubeSat, Kalman Filter, Neural Control, Reinforcement Learning, Attitude Estimation, Satellite Dynamics  

## Author
**Amaraneni Vinitha**  
B.Tech in Aeronautical Engineering | AI-based Space Systems Researcher  
GitHub: [@amaranenivinitha](https://github.com/amaranenivinitha)

## License
Released under the MIT License.
