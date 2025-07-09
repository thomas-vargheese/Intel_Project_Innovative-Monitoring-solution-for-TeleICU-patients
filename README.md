
# <span style="font-size:2em;">Innovative Monitoring System for TeleICU Patients Using Video Processing and Deep Learning</span>

## <span style="font-size:1.5em;">Introduction</span>

This project aims to develop an advanced monitoring system for TeleICU patients using video processing and deep learning techniques. The system is designed to enhance patient care by providing real-time monitoring, detecting critical events, and alerting healthcare professionals promptly.

## <span style="font-size:1.5em;">Features</span>

- **Real-time Video Processing**: Continuous monitoring of ICU patients through video feeds.
- **Motion Detection**: Identifies significant patient movements or changes in posture.
- **Deep Learning Models**: Utilizes state-of-the-art deep learning algorithms to analyze video data.

## <span style="font-size:1.5em;">Table of Contents</span>

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)

## <span style="font-size:1.5em;">Installation</span>
<span style="font-size:1.2em;">Install Dependencies</span>
```
pip install -r requirements.txt
```

### <span style="font-size:1.2em;">Prerequisites</span>

- Python 3.8 or higher
- pip (Python package installer)
- Operating System: Compatible with Windows, macOS, or Linux.
- Hardware: Depending on the scale of your project, a GPU may be recommended for faster training with TensorFlow (NVIDIA GPU with CUDA support).
- Software:Python environment set up with necessary packages (pip, virtualenv, or conda for managing Python environments).
- CUDA and cuDNN (if using GPU acceleration).
- IDE or text editor of choice for development (e.g., PyCharm, Visual Studio Code).

### <span style="font-size:1.5em;">Project Structure</span>
```
                                          Project Structure:
                                          
                                          object-detection/
                                          │
                                          ├── models/
                                          │   ├── teleicu_model.h5
                                          │   └── best.pt
                                          │
                                          ├── data/
                                          │   └── label_map.pbtxt
                                          │
                                          ├── videos/
                                          │   └── input_video.mp4
                                          │
                                          ├── teleicu_detection.py
                                          ├── yolov8_detection.py
                                          └── README.md
```
### <span style="font-size:1.2em;">Usage</span>
<span style="font-size:1.2em;">Running the System</span>
To start the monitoring system, run the following command:
```
python RealTimeMonitoring.py
```
<span style="font-size:1.2em;">Configuration</span>
You can configure various parameters such as video source, alert thresholds, and model settings in the config.json file.



### <span style="font-size:1.2em;">Clone the Repository</span>
```bash
git clone https://github.com/AleetaT/teleicu-monitoring-system.git
cd teleicu-monitoring-system
```
Link to the models
https://drive.google.com/drive/folders/1jDoQay2stf8z8tqW4VnM6FWYUaXdr1EF?usp=sharing

