# Object Tracking in VR with Meta Quest 2

## Table of Contents
1. [Overview](#overview)
2. [Features](#features)
3. [Hardware Requirements](#hardware-requirements)
4. [Setup and Installation](#setup-and-installation)
   - [Real-Life Setup](#real-life-setup)
   - [WebSocket Server Configuration](#websocket-server-configuration)
   - [Raspberry Pi Setup](#raspberry-pi-setup)
   - [Unity Project Setup](#unity-project-setup)
5. [Testing and Deployment](#testing-and-deployment)
   - [Testing](#testing)
   - [Deployment](#deployment)
6. [Troubleshooting](#troubleshooting)

---

## Overview

This project enhances the VR experience by integrating motion-tracked objects into a static VR environment. The system tracks objects using a camera, processes positional data on a Raspberry Pi, and streams it to a Unity application via a WebSocket server.

## Features

- **Real-time object tracking** using a camera.
- **Raspberry Pi integration** for efficient data processing.
- **WebSocket-based communication** for seamless data transfer.
- **Unity-based VR environment** designed for Meta Quest 2.
- **Support for multiple objects** with individual AR markers.

---

## Hardware Requirements

- **Camera**: Luxonis OAK-D Lite
- **Microprocessor**: Raspberry Pi 4 running Raspbian OS
- **VR Headset**: Meta Quest 2
- **Development Environment**: Computer with Unity installed and Meta Quest 2 Link setup.

---

## Setup and Installation

### Real-Life Setup
1. **Generate Markers**:
   - Visit [ArUco Marker Generator](https://chev.me/arucogen/) and print markers with IDs 1, 2, and 0.
   - Attach markers as follows:
     - **ID 1**: BOX1 in code.
     - **ID 2**: BOX2 in code.
     - **ID 0**: (Optional) Attach to your headset for user position tracking (not fully functional).
2. **Camera Placement**:
   - Position the camera on a stable, level surface (ideally a tripod) with a clear line of sight to all markers.
   - Ensure the area is well-lit with textured backgrounds to optimize tracking.
3. **Testing Setup**:
   - Avoid placing markers too close, as the system may struggle with detection.

---

### WebSocket Server Configuration
1. Locate the WebSocket server file in the project directory.
2. Update the **local IP address** in the server code to match your Raspberry Pi or network configuration.
3. Use `install_requirements.py` to install the necessary Python dependencies.
   - Ensure you do this within a Python virtual environment when working on the Raspberry Pi.

---

### Raspberry Pi Setup
1. Replace paths in `arUcoTrack.service` with the appropriate paths on your Raspberry Pi machine.
   - This service is used to run a headless version of `arucoTrackWeb.py` or `RPIArucoTrackWeb.py` on boot.
   - This ensures the system runs continuously as long as there’s a WiFi connection and the camera is connected properly.
2. Copy and enable the service with the following commands:
   ```bash
   sudo cp arUcoTrack.service /etc/systemd/system/
   sudo systemctl enable arUcoTrack.service
   sudo systemctl start arUcoTrack.service
3. For debugging or testing camera output, use the more verbose `arucoTrackWeb.py` script.
---

### Unity Project Setup
1. **Create Project**:
   - Use the Unity **VR Core Template** to create a new project.
2. **Add Interactable Objects**:
   - Take an interactable cube from the template environment and make a copy:
     - Name the first cube `BOX1`.
     - Name the second cube `BOX2`.
   - Disable Box Colliders if there to avoid collision issues between objects.
   - In the Rigid Body properties:
     - Increase drag and angular drag to above 1000 to reduce random flipping.
     - Disable gravity to prevent the objects from falling.
   - Ensure object names match those referenced in the WebSocket code.
3. **WebSocket Client Integration**:
   - Import the WebSocket client file into your Unity project.
   - Attach it to a new empty GameObject (e.g., `scriptRunner`) to run the script.
4. **Environment Configuration**:
   - Adjust the starting position of the player and environment as needed for optimal use.
   - Remove unnecessary objects from the scene and orient the player towards the origin (as the camera treats itself as position 0,0,0).

---

## Testing and Deployment

### Testing
1. Use **AirLink** or a **wired Link** to test the application.
2. Start the Unity project in **Play mode** to send the active session to the headset.
3. Stop the play session before making any code changes to avoid compiler errors.

---

### Deployment
1. **Switch Platform**:
   - Change the Unity project's build settings to **Android**.
2. **Build APK**:
   - Create an APK file from the Unity project.
3. **Install on Headset**:
   - Use the Unity Developer Hub or another APK installer to transfer the build to the Meta Quest 2.

*Note: Visible hand tracking might only work in APK deployment and not during testing.*

---

## Troubleshooting

### Common Issues
- **Object not detected**:
  - Ensure object names in Unity match the WebSocket client code.
  - Verify the camera has a clear view of the markers and is correctly configured.
- **Tracking inconsistencies**:
  - Check for proper lighting and ensure the camera remains stable.
- **WebSocket connection errors**:
  - Verify the correct IP address is configured in the server code.

---
