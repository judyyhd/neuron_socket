# Delivery Bot Smart Plug Analysis
[TOC]
# Statement of Purpose
This repository is the home to the socket data and analysis script. 
In this exploratory data analysis project, I was given two data files on two smart plugs deployed at a hotel. One of them is used by a automatic delivery bot, where the delivery person hand off the delivery to the bot, and the bot automatically goes to the designated hotel room to deliver the parcel. The second one is used by a public PC located in the hotel's lobby. 
The purpose of this project is to uncover any trend in the data, with which we could discover some insight into the operations of the delivery bot and/or that of the hotel. The hotel public PC data is intended as a control group. 

# About the Data
The socket data is provided by Bi-Fan Yin of 住友酒店, requested by Alex Hon. 
There are 4 columns in the original dataset: **equipmentName**, **functionType**, **time**, and **value**.
## equipmentName
Two unique values: Zprime_Socket_01 and Zprime_Socket_02. 01 corresponds to the smart plug used by automatic delivery bots, and 02 corresponds to the smart plug used by a public PC. 

## functionType
12 unique values: CSQ, Current, LastHourPF, Leakage, PartPF, PhaseAngle, Power, PowerFactor, RelayStatus, Temperature, TotalPF, and Voltage. 
According to the document provided by Yin (安驿电管家系列产品协议文档-V1.07 1.pdf), the values correspond to:
* CSQ: 信号强度百分比
  * 浮点数表示，0.00%-100.00%，百分比值越大，信号越强
* Current: 电流
  * 浮点数表示，单位安培（A）
* LastHourPF: 前一小时电能量
  * 浮点数表示，单位为度（KWH）
* Leakage: 漏电流
  * 浮点数表示，单位为毫安（mA）
* PartPF: 当日电能量
  * 浮点数表示，单位为度（KWH）
* PhaseAngle: 相位角
  * 浮点数表示，单位为度
* Power: 有功功率
  * 浮点数表示，单位瓦特（W）
* PowerFactor: 功率因数
  * 浮点数表示，单位无
* RelayStatus: 继电器状态
  * 0：继电器断开1：继电器闭合
* Temperature: 温度
  * 浮点数表示，单位为摄氏度（℃）
* TotalPF: 总电能量
  * 浮点数表示，单位为度（KWH）
* Voltage: 电压
  * 浮点数表示，单位伏特（V）

## time
Time at which the data was recorded. ISO 8601 format with a UTC offset. 

## value
The value of the specific functionType.

# Robot Activity Log Analysis Summary

This analysis explores patterns in a robot's operational data using time series processing, clustering, and state modeling. The goal is to characterize usage behavior and uncover temporal structure in the robot's activities.

## Steps Completed

1. **Preprocessing**
   - Parsed timestamps from raw logs.
   - Resampled data to 59-seconds intervals for uniform time series representation.

2. **Exploratory Data Analysis**
   - Computed correlation matrix between functions to understand relationships.
   - Visualized distributions of function values to identify distinct operational regimes.
   - Inspected time series plots to observe behavior dynamics.

3. **Clustering and Dimensionality Reduction**
   - Applied **K-Means clustering** to function time series and found 3 interpretable clusters.
   - Used **Principal Component Analysis (PCA)** to confirm that **PC1** captures overall robot activity.
   - Interpreted cluster centroids and PCA loadings to assign functional meanings to clusters:
     - `Idle`: Robot is inactive and not charging.
     - `Active charging`: Robot is plugged in.
     - `Moderate`: Robot is performing regular tasks.

4. **State Assignment and Transition Modeling**
   - Labeled time series data with assigned states from clustering.
   - Computed total time spent in each state.
   - Built a **Markov transition matrix** and computed the **stationary distribution** of robot states.

5. **Temporal Usage Profiling**
   - Aggregated state durations by:
     - **Day of week**
     - **Time of day** (morning / afternoon / evening / night)
     - **2-hour intervals**
   - Visualized state proportions using grouped bar plots to reveal patterns in state usage across time.