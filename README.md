# Intelligent Accident Detection and Alert System

## Abstract
This repository contains the Intelligent Accident Detection and Alert System, designed to enhance urban mobility and safety by leveraging CCTV footage and video stream analytics for the accurate detection of car collisions. This system employs advanced machine learning models, including a transfer-learning model for accident detection and YOLOv5 for human detection, to trigger alarms efficiently in the absence of human presence at accident scenes.

## Project Overview
The project aims to reduce emergency response times and mitigate the impact of road accidents through intelligent automation and precise analytics. It was developed using a combination of Python, OpenCV for video processing, and TensorFlow for implementing the machine learning models.

## Key Features
Accident Detection: Utilizes a transfer-learning approach with pre-trained MobileNetV2 to detect vehicular collisions in real-time.
Human Detection: Integrates YOLOv5 to identify human presence post-accident, ensuring alarms are triggered only when necessary.
Alarm System: Triggers an automated alert system if no human is detected within a specified timeframe after an accident.


## Installation
Prerequisites
Python 3.8 or higher
pip (Python package installer)

Dependencies
Install the necessary Python packages using pip:

bash
pip install -r requirements.txt


## Setup
git clone git clone https://github.com/boxchan/Accident_Detection.git


## Usage
To start the accident detection system, run the following command from the root of the project:
python process_detect.py

## Author
Yingzhu Zhang
Jiaming Bai
Suchan Park


Special thanks to the data providers and academic advisors.
