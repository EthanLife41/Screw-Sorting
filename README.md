# Screw Detection Prototype
Praxis II Project - Screw Organizing for GTA Woodworks

## Overview

Mid-fidelity Python prototype that detects when a screw enters a defined region in a camera feed, captures an image, and uses OpenAI to verify if a screw is present. Based on the result, it outputs an action.

## Feature
* OpenCV for detection
* Learns an empty background before detecting objects
* Triggers capture when a new object enters the ROI
* Crops and saves only the inspection region (`captures/latest_capture.jpg`)
* Uses OpenAI to return **YES/NO** for screw presence
* Displays a result screen with action and countdown
* Supports camera switching and reconnects on failure 

## Controls
* `q` quit
* `c` manual capture
* `n` next camera (connected to laptop)
* `p` previous camera (connected to laptop)
* `r` reset detection

## Setup
1. Dependencies: OpenCV, NumPy, python-dotenv, openai
2. Add `.env` file:
```env
OPENAI_API_KEY=your_key_here
```

3. Run the script

## Notes
* Only detects presence, not screw type (for now)
* Hands or large objects in the box may affect results
* Each capture overwrites the previous image 
