# Day 03 - YOLO Helmet / No Helmet Detection

## Overview

Day 03 focused on training and testing a YOLO-based helmet detection model using a custom helmet / no-helmet dataset.

The model was trained to classify two safety-related classes:

- `helmet`
- `nonhelmet`

This experiment is part of an AI-based workplace safety monitoring project using computer vision.

## Project Goal

The goal of this stage was to train a lightweight object detection model that can recognize whether a person is wearing a safety helmet.

This can be used as a basic safety monitoring function in small industrial workplaces where workers may approach machines or hazardous zones without proper protective equipment.

## Folder Structure

```text
day03-yolo-helmet-detection/
├── README.md
├── models/
│   └── best.pt
├── demo/
│   └── helmet_detection_demo.mp4
├── data/
│   └── helmet_nonhelmet_dataset.zip
├── logs/
│   └── training_logs.zip
└── results/
    └── yolo_fps.csv
