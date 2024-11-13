# Computer Vision-based Traffic Violation Detection System

**Student ID**: 100129  
**Name**: Janisha Hota  
**School**: Delhi Public School, Bangalore North  
**Email**: janishahota@gmail.com  

---

## Introduction
Computer vision is a field within artificial intelligence (AI) that leverages machine learning and neural networks to enable systems to extract meaningful insights from images, videos, and other visual data. Computer vision is increasingly used in applications such as robotics, manufacturing, and healthcare.

---

## Problem Statement
**"Road Rules Are Not Suggestions – They’re Lifesavers."** Following traffic rules is essential to prevent accidents and save lives. In India, official statistics from the Ministry of Road Transport and Highways (MoRTH) reported 153,972 road fatalities in 2021, equating to 11.3 deaths per 100,000 people. With the rise in private vehicle usage, the workload on traffic police has increased significantly. Enforcing penalties for traffic violations can encourage safer driving behaviors and improve road safety.

---

## Goal and Objective
To develop an AI-powered traffic violation detection model capable of identifying:
- Red light running
- Triple riding on two-wheelers
- Using a phone while driving
- Helmet use violations on two-wheelers

---

## Approach

### Software Development Life Cycle (SDLC) Model
I employed the traditional waterfall SDLC model, which follows a top-down approach through different stages.

1. **Planning**  
   Conducted a survey to understand public perceptions of traffic violation detection and common types of violations.  
   - Survey Link: [Google Form Survey](https://forms.gle/ojh6PLXofioks5Te9)  

2. **Defining**  
   Compiled necessary resources, including datasets for:
   - Car (for red light running)
   - Motorcycle/Two-wheeler (for helmet and triples detection)
   - Traffic Light (for red light running)
   - Phone and Steering Wheel (for phone use detection)

3. **Designing**  
   Created a project storyboard to outline the ideal user experience.  

4. **Development**  
   Using Pycharm and Python, I divided the code into parts for each violation type. Below are the steps for each violation model:
   
   - **Red Light Running**  
      - Loaded YOLOv8 model to detect "traffic light" and "car" objects.
      - Defined a violation zone and detected red lights using HSV color space.
   
   - **Phone Usage While Driving**  
      - Used two YOLO models for cell phone and steering wheel detection.
      - Calculated detection accuracy by counting frames with both objects detected.
   
   - **Helmet Detection**  
      - Loaded YOLOv8 model to detect "helmet" and "without helmet" classes.
      - Sorted and displayed frames with helmet violations.
   
   - **Triples Detection**  
      - Detected two-wheelers and people within specified distances to flag triples riding violations.

5. **Testing and Deployment**  
   Each violation model was tested independently to validate detection accuracy:
   - No Helmet Detection: 85%
   - Phone Usage While Driving: 74%
   - Red Light Running: 94%
   - Triples Detection: 78.99%

---

## Limitations
- **Model Accuracy in Varied Environments**: YOLOv8 may have reduced accuracy in low-light or adverse weather conditions.
- **False Positives/Negatives**: Occasional misclassification may lead to false alerts or missed violations.
- **Real-time Processing**: Processing may be delayed, affecting real-time monitoring capability on standard hardware.

---

## Future Scope
- **Real-time Optimization**: Implement model compression for real-time monitoring.
- **License Detection Integration**: Add license plate recognition to track vehicles and automate fines for violations.

---

## Results and Conclusion
Through this project, I gained practical insights into the SDLC process and experienced the challenges of developing computer vision models for real-world applications. This project highlights the potential of AI in enhancing traffic safety and the real-life implications of computer vision in societal safety measures.

---

## References
1. [IIT Delhi - Road Safety](https://tripc.iitd.ac.in/assets/publication/RSI_2023_web.pdf)
2. [IBM - Computer Vision](https://www.ibm.com/topics/computer-vision)
3. [JavaTpoint - Computer Vision Applications](https://www.javatpoint.com/computer-vision-applications)
4. Clark, H. *The Software Development Life Cycle (SDLC): 7 Phases and 5 Models*. The Product Manager. Retrieved January 15, 2024, from [The Product Manager](https://theproductmanager.com/topics/software-development-life-cycle/)
5. [Infinity Learn - Traffic Rules](https://infinitylearn.com/surge/english/slogans/slogans-on-traffic-rules/#:~:text=%E2%80%9CRoad%20Rules%20Are%20Not%20Suggestions,%2C%20Red%20Light%20for%20Risk.%E2%80%9D)
