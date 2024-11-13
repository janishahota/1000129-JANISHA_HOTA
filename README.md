# Machine Learning and Deep Learning

## Developing a Computer Vision-based Detection System using Machine Learning
## Scenario 1:Computer Vision-based Traffic Violation Detection System

**Student ID**: 1000129  
**Name**: Janisha Hota  
**School**: Delhi Public School Bangalore North  
**Email**: janishahota@gmail.com  

---

## Introduction:
Computer vision is a field of Artificial Intelligence (AI) that uses machine learning and neural networks to teach computers and systems to derive meaningful information from digital images, videos and other visual inputs—and to make recommendations or take actions when they see defects or issues. (2) Nowadays, Computer vision is in great demand and used in different areas, including robotics, manufacturing, healthcare, etc. (3)

---

## Problem Statement:
Road Rules Are Not Suggestions – They’re Lifesavers. (5) Following traffic rules and safety guidelines is of the utmost importance to avoid life-threatening accidents from occurring. According to official statistics published by the Ministry of Road Transport and Highways (MoRTH), 153,972 persons were killed in road crashes in the year 2021 in India. This corresponds to 11.3 deaths per 100,000 population. (1) With enforced regulations and fines to discourage irresponsible driving, this rate will decrease. Due to the rise in private vehicles in India over the past few years, the workload on the traffic police has greatly increased too. In case of any incidents or traffic rules violations, there should be repercussions for the perpetrator, to ensure safety and mindfulness on roads.

---

## Goal and Objective:
To create an AI Traffic violation detection model that can detect red light running and using a phone while driving for a four-wheeler, and helmet detection and triples riding for a two-wheeler using computer vision and machine learning.

---

## Approach:
I will be following the traditional waterfall SDLC model, which consists of a top-down approach while following the stages of the Software Development Life Cycle (SDLC) (Clark, n.d.).

1. **Planning**:  
   I first fine-tuned my idea. I surveyed to understand people's opinions on traffic violation detection and which violations they’ve witnessed occur frequently.  
   - Survey link: [Google Form Survey](https://forms.gle/ojh6PLXofioks5Te9)
   - Survey responses: [Survey responses](https://docs.google.com/spreadsheets/d/1_9gj4T2OagWgtnQu7rOWU9C9fk4x3WrNTeonmEzq8aM/edit?usp=sharing)
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
   ![Storyboard (1)](https://github.com/user-attachments/assets/439c23eb-5898-4d58-95b9-774b796d71b9)

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
        <img width="709" alt="Screenshot 2024-11-13 at 9 09 32 AM" src="https://github.com/user-attachments/assets/d5dd23d9-c888-45bc-bbf6-d9d9bc9d3ed3">
      - Processed each video frame with YOLO, filtered detections by confidence, and checked if a car in the violation zone coincided with a red light.
      - Drew bounding boxes on detected objects and the violation zone for clarity.
      - Calculated detection accuracy based on true and false positives and displayed frames with detected violations.
      <img width="500" alt="Screenshot 2024-11-13 at 9 10 34 AM" src="https://github.com/user-attachments/assets/eceb577b-3c61-423b-9806-558c8567b40a">

   - **Phone Usage while Driving**:  
      - Make a custom model using a labelled dataset from Roboflow to ensure more accuracy for cell phone detection.
      <img width="840" alt="Screenshot 2024-11-13 at 9 16 28 AM" src="https://github.com/user-attachments/assets/fbee1469-97e7-4ab5-b680-5ccebb848c95">
      -Loaded two YOLO models:  a custom model for detecting cell phones and yolo for detecting steering wheels.
        <img width="1008" alt="Screenshot 2024-11-13 at 9 10 58 AM" src="https://github.com/user-attachments/assets/f071201d-14be-4e4c-8683-f94085741db1">
      - Opened the video file and initialised counters for total frames and frames with both phone and wheel detected.
      - For each frame, I ran the phone model and the wheel model separately.
      - Checked if a detected object was a cell phone (green box) or a wheel (blue box) and drew bounding boxes with labels.
        <img width="919" alt="Screenshot 2024-11-13 at 9 11 35 AM" src="https://github.com/user-attachments/assets/3577e305-8ab8-4f72-81e1-3bc980571f8b">
      - If both objects were detected, counted the frame and updated the highest confidence frame if the current confidence was greater.
      - Displayed each frame with bounding boxes in real-time.
      - After processing, the accuracy as the percentage of frames where both objects were detected.
      - Displayed the best frame (highest confidence) if a violation was detected.
        <img width="393" alt="Screenshot 2024-11-13 at 9 14 50 AM" src="https://github.com/user-attachments/assets/08a54a61-6f52-4f51-859a-1fef9835f96f">


   - **Helmet Detection on a 2-wheeler**:  
      -Make a custom model, by obtaining a labelled dataset from Roboflow.
     <img width="789" alt="Screenshot 2024-11-13 at 9 18 54 AM" src="https://github.com/user-attachments/assets/e734afb1-b70d-4f72-88fc-7c2df3a9b120">
      - Load  the custom YOLOv8 model for detecting the "helmet" and "without helmet" classes.
      - <img width="535" alt="Screenshot 2024-11-13 at 9 15 20 AM" src="https://github.com/user-attachments/assets/eab251be-1a2e-43a9-a75f-c4814ad9ce4c">
      - Open video file, initialise a list to track violation frames and total frames.
      - Loop through video frames:
        - Run model inference on each frame.
        - Identify objects as "helmet" or "without helmet" based on detections.
        - For "without helmet":
          - Record frame and confidence score.
          - Draw red bounding box with label.
       <img width="1259" alt="Screenshot 2024-11-13 at 9 19 44 AM" src="https://github.com/user-attachments/assets/0b75bd23-32a8-48f8-91b5-26274d35220f">
        - Display each frame with bounding boxes.
      - After processing, sort violation frames by confidence.
      - Display each top violation frame.
<img width="395" alt="Screenshot 2024-11-13 at 9 20 28 AM" src="https://github.com/user-attachments/assets/bda22784-57ce-4bff-8bcd-31895bcae604">

   - **Triples Detection**:  
      - Load YOLOv8 model and open the video file.
      - Initialize counters for frames, violations, and detections.
      - Loop through each frame, running YOLO on it to detect objects.
      - For each detection:
        - Check if the object is a two-wheeler or a person.
        - Store bounding box coordinates and draw them with color-coded labels.
          <img width="500" alt="Screenshot 2024-11-13 at 9 20 42 AM" src="https://github.com/user-attachments/assets/7315d4fd-17fd-4642-bb3f-567823acb825">
      - For each detected two-wheeler:
        - Calculate distances between detected people.
        - If three or more people are within a certain distance, flag a violation and store the frame.
          <img width="389" alt="Screenshot 2024-11-13 at 9 21 17 AM" src="https://github.com/user-attachments/assets/6a24b596-5cea-406c-ba6f-7d2af151c927">
      - Calculate detection accuracy as average detections per frame.
      - Display each frame with bounding boxes and accuracy info.
      - Store and display frames where violations were detected.
      - Summarise results: total frames, violation count, and final detection accuracy.
      <img width="240" alt="Screenshot 2024-11-13 at 9 21 56 AM" src="https://github.com/user-attachments/assets/4817b460-bc02-4964-a0a2-e601e3d86451">


5. **Testing and Deployment**:  
   Each violation detection model was tested independently to validate that the bounding boxes and violation conditions were functioning correctly. For example, I confirmed that the red-light-running detector accurately recognized cars crossing a designated boundary while the light was red, and similarly, that helmet and phone detection flagged violations accurately. The Accuracy rate for each of the models were:  
   - No Helmet Detection: 85%  <img width="200" alt="Screenshot 2024-11-13 at 9 22 10 AM" src="https://github.com/user-attachments/assets/37fc8ee5-ec65-4034-9815-d5521996ce70">

   - Phone Usage while driving: 74%  <img width="200" alt="Screenshot 2024-11-13 at 9 22 21 AM" src="https://github.com/user-attachments/assets/efc3b407-4caf-49d2-9da7-534798c2ba5f">

   - Red-light running: 94%  <img width="200" alt="Screenshot 2024-11-13 at 9 22 32 AM" src="https://github.com/user-attachments/assets/6319a98c-6d27-4427-913e-3dfab9efbb0c">

   - Triples Detection: 78.99%   <img width="200" alt="Screenshot 2024-11-13 at 9 22 50 AM" src="https://github.com/user-attachments/assets/2a0ab59f-dca3-4bcc-9ee9-d59e2e3b9a5f">

---

## Limitations:
- **Model Accuracy in Varied Environments**: The YOLOv8 model may perform inconsistently in different lighting conditions, weather, or complex backgrounds. For example, poor lighting or rainy conditions may reduce detection accuracy for objects like helmets, traffic lights, or two-wheelers.
- **False Positives and False Negatives**: Despite a high confidence threshold, the model may occasionally misclassify objects, such as detecting non-helmet objects as helmets or failing to detect a person on a two-wheeler. This impacts the reliability of violation detection, leading to false alerts or missed violations.
- **Detection Speed and Real-time Processing**: Running YOLOv8 on video or real-time feeds may cause processing delays, especially on standard hardware, limiting its applicability for real-time enforcement or monitoring.

---

## Future Scope:
- **Real-time Optimization**: Improving model efficiency through model compression can enable real-time violation detection, making it feasible for live traffic monitoring and enforcement.
- **Integration with License Detection and Tracking**: With car license plate detection, we can track which vehicles have done a violation and send fines to the registered vehicle's owner.

---

## Results and Conclusion:
By developing this traffic violation model, I've had many key takeaways. By implementing the SDLC process, I've experienced what it's like to be in the shoes of a developer and learned firsthand how it might be as a profession in the future. This project also gave me the opportunity to understand Yolo and implement it along with computer vision in real-life scenarios.

---

## References:
- https://tripc.iitd.ac.in/assets/publication/RSI_2023_web.pdf  
- https://www.ibm.com/topics/computer-vision  
- https://www.javatpoint.com/computer-vision-applications  
- Clark, H. (n.d.). *The Software Development Life Cycle (SDLC): 7 Phases and 5 Models*. The Product Manager. Retrieved January 15, 2024, from https://theproductmanager.com/topics/software-development-life-cycle/  
- https://infinitylearn.com/surge/english/slogans/slogans-on-traffic-rules/#:~:text=%E2%80%9CRoad%20Rules%20Are%20Not%20Suggestions,%2C%20Red%20Light%20for%20Risk
