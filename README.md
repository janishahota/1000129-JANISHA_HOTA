# Computer Vision-based Traffic Violation Detection System

**Student ID**: 100129  
**Name**: Janisha Hota  
**School**: Delhi Public School, Bangalore North  
**Email**: janishahota@gmail.com  

---

## Introduction:
Computer vision is a field of artificial intelligence (AI) that uses machine learning and neural networks to teach computers and systems to derive meaningful information from digital images, videos and other visual inputs—and to make recommendations or take actions when they see defects or issues. (2) Nowadays, Computer vision is in great demand and used in different areas, including robotics, manufacturing, healthcare, etc. (3)

---

## Problem Statement:
Road Rules Are Not Suggestions – They’re Lifesavers. (5) Following traffic rules and safety guidelines is of the utmost importance to avoid life-threatening accidents from occurring. According to official statistics published by the Ministry of Road Transport and Highways (MoRTH), 153,972 persons were killed in road crashes in the year 2021 in India. This corresponds to 11.3 deaths per 100,000 population. (1) With enforced regulations and fines to discourage irresponsible driving, this rate will decrease. Due to the rise in private vehicles in India over the past few years due to the pandemic, the workload on the traffic police has greatly increased too. In case of any incidents or traffic rules violations, there should be repercussions for the perpetrator, to ensure safety and mindfulness on roads.

---

## Goal and Objective:
To create an AI Traffic violation detection model that can detect red light running, triples riding, using a phone while driving and helmet detection on a two-wheeler using computer vision and machine learning.

---

## Approach:
I will be following the traditional waterfall SDLC model, which consists of a top-down approach while following the stages of the Software Development Life Cycle (SDLC) (Clark, n.d.).

1. **Planning**:  
   I first fine-tuned my idea. I surveyed to understand people's opinions on traffic violation detection and which violations they’ve witnessed occur frequently.  
   - Survey link: [Google Form Survey](https://forms.gle/ojh6PLXofioks5Te9)  
   - Survey responses:  
   I researched various current technological solutions for traffic violation detection, such as the e-Challan system in Karnataka, where violations are detected in real-time and fines are given to the vehicle’s registered owner. I decided to make a similar basic model for a few violations, based on the most occurring violations witnessed, which I obtained through my survey responses. They were red-light running, 2-wheeler Helmet detection and triples detection, and the usage of phones while driving.

2. **Defining**:  
   In this stage, I planned for what resources I would require, and how the app could be utilised. The resources required would be datasets for the following:  
   - Car (for red-light running)  
   - Motorcycle/2-wheeler (for triples detection and no helmet detection)  
   - Helmet/No Helmet (for no helmet detection)  
   - Traffic Light (for red-light running)  
   - Phone (for Phone usage while driving)  
   - Steering Wheel (for Phone usage while driving)  
   - People (for triples detection)

3. **Designing**:  
   In this stage, I turn my plan into a framework for the project. I have developed an ideal user experience storyboard.
   
   **[Storyboard made using canva.com]**

4. **Development/Building**:  
   I used Pycharm as the platform for creating the Python code.  
   **Libraries used**:  
   - Subprocess  
   - Yolo  
   - Matplotlib  
   - Cv2  
   - Numpy

   I started by splitting the code into separate parts based on the violation.  
   
   - **Red Light Running**:  
      - Loaded YOLOv8 model to detect “traffic light” and “car.”
      - Defined a vertical violation zone where car positions are checked.
      - Created a function to detect red traffic lights using HSV color space.
      - Processed each video frame with YOLO, filtered detections by confidence, and checked if a car in the violation zone coincided with a red light.
      - Drew bounding boxes on detected objects and the violation zone for clarity.
      - Calculated detection accuracy based on true and false positives and displayed frames with detected violations.

   - **Phone Usage while Driving**:  
      - Loaded two YOLO models: one for detecting cell phones and a custom model for detecting steering wheels.
      - Opened the video file and initialised counters for total frames and frames with both phone and wheel detected.
      - For each frame, I ran the phone model and the wheel model separately.
      - Checked if a detected object was a cell phone (green box) or a wheel (blue box) and drew bounding boxes with labels.
      - If both objects were detected, counted the frame and updated the highest confidence frame if the current confidence was greater.
      - Displayed each frame with bounding boxes in real-time.
      - After processing, the accuracy as the percentage of frames where both objects were detected.
      - Displayed the best frame (highest confidence) if a violation was detected.

   - **Helmet Detection on a 2-wheeler**:  
      - Load YOLOv8 model for detecting the "helmet" and "without helmet" classes.
      - Open video file, initialise a list to track violation frames and total frames.
      - Loop through video frames:
        - Run model inference on each frame.
        - Identify objects as "helmet" o
