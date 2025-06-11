# neuron_socket
[TOC]
## Statement of Purpose
This repository is the home to the socket data and analysis script. 
In this exploratory data analysis project, I was given two data files on two smart plugs deployed at a hotel. One of them is used by a automatic delivery bot, where the delivery person hand off the delivery to the bot, and the bot automatically goes to the designated hotel room to deliver the parcel. The second one is used by a public PC located in the hotel's lobby. 
The purpose of this project is to uncover any trend in the data, with which we could discover some insight into the operations of the delivery bot and/or that of the hotel. The hotel public PC data is intended as a control group. 

## About the Data
The socket data is provided by Bi-Fan Yin of 住友酒店, requested by Alex Hon. 
There are 4 columns in the original dataset: **equipmentName**, **functionType**, **time**, and **value**.
### equipmentName
Two unique values: Zprime_Socket_01 and Zprime_Socket_02. 01 corresponds to the smart plug used by automatic delivery bots, and 02 corresponds to the smart plug used by a public PC. 

### functionType
Four unique values: CSQ, Current, LastHourPF, and Other. 
According to the document provided by Yin (安驿电管家系列产品协议文档-V1.07 1.pdf), the values correspond to:
* CSQ: 信号强度百分比
  * 浮点数表示，0.00%-100.00%，百分比值越大，信号越强
* Current: 电流
  * 浮点数表示，单位安培（A）
* LastHourPF: 前一小时电能量
  * 浮点数表示，单位为度（KWH）

### time
Time at which the data was recorded. ISO 8601 format with a UTC offset. 

### value
The value of the specific functionType.

## About the Script