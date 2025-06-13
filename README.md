# Smart Plug Analysis for Hotel Devices
[TOC]

## Statement of Purpose
This repository contains the data and analysis scripts for an exploratory study of two smart plugs deployed in a hotel setting. The first smart plug powers an **automated delivery robot**, which transports parcels from staff to hotel guests. The second powers a **public PC** located in the hotel’s lobby.

The project’s aim is to uncover operational patterns and energy usage trends, which can provide insights into the behavior and utilization of each device. By comparing these patterns, we explore how energy consumption reflects different usage contexts (robotic automation vs. human-operated terminal) and identify opportunities for smart facility optimization.

## About the Data
The socket data is provided by Bi-Fan Yin of 住友酒店 and was requested by Alex Hon.

Each dataset includes four columns:

- **equipmentName**: 
  - `Zprime_Socket_01` — delivery robot plug  
  - `Zprime_Socket_02` — public PC plug
- **functionType**: 12 types, including current, power, voltage, relay status, temperature, and more (detailed below).
- **time**: ISO 8601 timestamp with UTC offset.
- **value**: The measured reading for the specific functionType.

### FunctionType Dictionary
| FunctionType    | Description                        | Unit            |
|----------------|------------------------------------|-----------------|
| CSQ            | Signal strength (%)                | %               |
| Current        | Electrical current                 | A               |
| LastHourPF     | Energy in previous hour            | kWh             |
| Leakage        | Leakage current                    | mA              |
| PartPF         | Energy today                       | kWh             |
| PhaseAngle     | Phase angle                        | °               |
| Power          | Active power                       | W               |
| PowerFactor    | Power factor                       | (unitless)      |
| RelayStatus    | Relay status (0=off, 1=on)         | binary          |
| Temperature    | Device temperature                 | °C              |
| TotalPF        | Total energy                       | kWh             |
| Voltage        | Voltage                            | V               |

---

## Report Structure

The full analysis report is structured into five parts:

1. **Introduction**  
   Overview of the project goal, context, and data structure.
2. **Delivery Bot Analysis**  
   Characterization of the bot's operation via time series analysis, clustering, and state modeling.
3. **Public PC Analysis**  
   Similar methods applied to the PC dataset as a behavioral contrast.
4. **Comparative Insights**  
   A discussion of how the robot and PC differ in terms of energy patterns, routines, and clustering structure.
5. **Conclusion**  
   Summarizes actionable insights and how the identified data structure could support real-time monitoring, optimization, or automation strategies.

---

## Analysis Highlights

### 📦 Delivery Bot Smart Plug

- **Preprocessing**: Timestamps parsed, time series resampled to consistent intervals.
- **Feature Exploration**: Correlation matrix, value distributions, and trend visualizations.
- **Clustering**: PCA + K-Means clustering + manual boundary tuning.
- **State Modeling**: Custom state labels (e.g., idle, charging, active), transition matrices, stationary distributions.
- **Temporal Profiling**: Usage breakdowns by hour, weekday, and time-of-day segments.

### 🖥️ Public PC Smart Plug

- Similar approach applied to uncover usage routines.
- Clustering more effective due to less binary structure.
- Power states captured weekday vs. weekend activity.
- PCA revealed key drivers like ambient temperature and voltage influence.

### 📊 Comparison
- The delivery bot’s usage is **highly polarized**, dominated by predictable charging cycles.
- The public PC has **more variable and human-driven usage**, enabling richer unsupervised clustering.
- Analysis framework scales across devices and reveals potential **automation thresholds** and **scheduling opportunities**.

---

## Repository Contents

- `data/`: Raw and cleaned datasets.
- `notebooks/`: Jupyter notebooks used for analysis.
- `figures/`: Visualizations generated during the project.
- `report/`: Compiled LaTeX report and output PDF.
- `README.md`: Project overview.

---

## Citation
If you reference or build upon this work, please cite the GitHub repo or contact Judy Yang.