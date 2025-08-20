# Traffic-Flow-Analysis

### Traffic Lane Vehicle Counter (YOLOv8 + ByteTrack)

This project analyzes traffic flow by counting vehicles in **3 lanes** using **YOLOv8** detection and **ByteTrack** tracking.  
It overlays real-time counts on the video and logs detailed results into a CSV file.

---

## ✨ Features
- Vehicle detection (car, bus, truck, motorbike) using YOLOv8n (COCO pre-trained model)
  
- Automatic 3-lane division of the video frame
  
- ByteTrack algorithm for stable vehicle tracking IDs
  
- Prevents duplicate counting across frames
  
- Exports:
- 
  - Annotated video with lane overlays and live counts
    
  - CSV file with `[vehicle_id, lane, frame, timestamp_sec]`
    
- Console summary with per-lane totals

---

## ⚙️ Setup Instructions

### 1. Clone the repository

git clone https://github.com/<your-username>/traffic-lane-counter.git

cd traffic-lane-counter


### 2. Create a virtual environment (recommended)

python -m venv .venv

source .venv/bin/activate    # Linux / Mac

.venv\Scripts\activate       # Windows


### 3. Install dependencies

   
pip install -r requirements.txt

#### Dependencies include:

ultralytics (YOLOv8)

opencv-python

torch

numpy

pandas

### ▶️ Execution


Run the system

python main.py --video path/to/input.mp4 --output output/


Arguments:

--video : Path to input video file (local .mp4 file)

--output : Directory where results will be saved (default: output/)

## 📂 Outputs

##### Annotated Video

Saved as: output/traffic_out.mp4

Contains lane overlays and live vehicle counts

##### CSV File

Saved as: output/counts.csv

Columns: vehicle_id, lane, frame, timestamp_sec

##### Console Summary

=== SUMMARY ===
Lane 1: 659

Lane 2: 595

Lane 3: 49

Total vehicles: 1303


## 📖 Project Structure


traffic-lane-counter/

├─ main.py                  # Main script (detection + tracking + counting)

├─ requirements.txt         # Python dependencies

├─ README.md                # Setup & execution guide

├─ LICENSE                  # MIT License

├─ .gitignore

├─ docs/
│   └─ ARCHITECTURE.md      # High-level architecture

└─ output/

    └─ DEMO_VIDEO_LINK.txt  # Paste demo link here
    

## 📜 License
This project is licensed under the MIT License.


