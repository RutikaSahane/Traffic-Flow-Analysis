## Traffic Lane Vehicle Counter (YOLOv8 + ByteTrack)

This project analyzes traffic flow by counting vehicles in **3 lanes** using **YOLOv8** for detection and **ByteTrack** for tracking. 
It overlays real-time counts on the video and logs results into a CSV file for further analysis.

---

## ✨ Features
- Detects vehicles: **car, bus, truck, motorbike** using YOLOv8n (COCO pre-trained model)
  
- Automatic 3-lane division of video frames

- Stable vehicle tracking with **ByteTrack**
  
- Prevents duplicate counting across frames
  
- Outputs:
  
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

#### Activate the environment

source .venv/bin/activate    # Linux / Mac

.venv\Scripts\activate       # Windows

### 3. Install dependencies

pip install -r requirements.txt

Dependencies include: ultralytics, opencv-python, torch, numpy, pandas

▶️ Running the Project

python main.py --video path/to/input.mp4 --output output

Arguments:

--video : Path to the input video (.mp4)

--output : Directory to save results (default: output/)

## 📂 Outputs

##### Annotated Video

Saved as output/traffic_out.mp4

Shows lane overlays and live vehicle counts

##### CSV File

Saved as output/counts.csv

Columns: vehicle_id, lane, frame, timestamp_sec

##### Console Summary

=== SUMMARY ===

Lane 1: 659

Lane 2: 595

Lane 3: 49

Total vehicles: 1303


## 📖 Project Structure

traffic-lane-counter/

├─ main.py                  # Main detection + tracking script

├─ requirements.txt         # Python dependencies

├─ README.md                # Setup & execution guide

├─ LICENSE                  # MIT License
├─ .gitignore

├─ docs/

│   └─ ARCHITECTURE.md      # High-level project architecture

└─ output/

    └─ https://drive.google.com/drive/folders/1T42xqYU2y21nClbQymh6EchRv-esL3rA?usp=drive_link
    
## 📜 License 

This project is licensed under the MIT License.


