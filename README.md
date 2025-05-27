# IntelliStep – A Smart Stick for the Blind

## Demo
Watch the demo on YouTube: [https://youtu.be/TimTm1kJ488](https://youtu.be/TimTm1kJ488)

## Overview

**IntelliStep** is an advanced assistive smart stick built to improve mobility and independence for visually impaired individuals. By combining a range of sensors with smart processing units, the system offers real-time audio feedback to guide users safely through their surroundings.

## Components Used

- **Arduino Nano** – Manages input from ultrasonic sensors and controls the buzzer for obstacle alerts.
- **ESP32** – Acts as the central processor, handling camera input, object detection, and Bluetooth audio output.
- **Neo-6M GPS Module** – Provides current location data for navigation and potential emergency use.
- **Ultrasonic Sensors** – Identify nearby obstacles and trigger alerts.
- **OV Camera Module** – Captures images for real-time object detection.
- **YOLOv5** – A lightweight, real-time object detection model used to recognize and classify objects in the environment.
- **Bluetooth Earphones** – Deliver verbal feedback to the user without obstructing environmental sounds.
- **Buzzer** – Emits audio alerts when an obstacle is detected.

## How It Works

1. **Obstacle Detection**: Ultrasonic sensors detect obstacles in close range. If an object is too near, the buzzer provides a warning.
2. **Object Detection**: The OV camera sends visual data to the ESP32, where YOLOv5 identifies and classifies nearby objects.
3. **Location Tracking**: The Neo-6M GPS module fetches real-time coordinates for navigation or emergency communication.
4. **Audio Feedback**: The ESP32 transmits spoken messages via Bluetooth to the earphones, informing the user about obstacles, object types, and GPS location when needed.

## Prototype vs Final Version

- The **prototype** utilizes the Arduino Nano for sensor handling and ESP32 for processing and Bluetooth communication. Power is currently drawn from the Arduino’s 5V output.
- The **final version** will include a dedicated battery for stable power and a **SIM800 module** to provide cellular connectivity, enabling enhanced features like emergency communication.

## Key Features

- Real-time **obstacle detection** and buzzer alerts.
- **Smart object recognition** using YOLOv5 deep learning model.
- **GPS location tracking** for navigation and safety.
- **Audio feedback** through Bluetooth earphones.
- Planned **GSM module integration** for mobile network access.

## Applications

- Designed to aid blind and visually impaired individuals in daily movement.
- Useful in both urban and semi-urban areas.
- Potential for future integration with alert systems and caregiver communication tools.

## Future Enhancements

- Add GSM/GPRS communication using the SIM800 module.
- Optimize power usage with efficient battery management.
- Expand object detection dataset for more accurate results.
- Develop a companion mobile app for real-time monitoring and assistance.
