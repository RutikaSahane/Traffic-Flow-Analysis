# Traffic Lane Vehicle Counter (YOLOv8 + ByteTrack)

This project analyzes traffic flow by counting vehicles in **3 lanes** using **YOLOv8** detection and **ByteTrack** tracking.

## Features
- Vehicle detection (car, bus, truck, motorbike)
- Automatic 3-lane division
- ByteTrack for stable tracking IDs
- Exports:
  - Annotated video with overlays
  - CSV with [vehicle_id, lane, frame, timestamp_sec]

## Quickstart
```bash
pip install -r requirements.txt
python main.py --video path/to/input.mp4 --output output/
```

## Output
- `output/traffic_out.mp4` – annotated video
- `output/counts.csv` – vehicle count records
- Console summary: per-lane totals

## Demo
- Record 1–2 minutes while the system runs.
- Upload demo to Google Drive or GitHub release.
- Paste link in `output/DEMO_VIDEO_LINK.txt`.
