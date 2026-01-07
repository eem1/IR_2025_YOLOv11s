data prep for training yolo11 IR detection model

Folder Structure 

C:\seals_project\          <-- Main Project Folder
│
├── train.py               <-- Your script
├── data.yaml              <-- Your config
├── ice_seals...csv
│
├── images\                <-- NEW: Move all .png files here
│   ├── image_01.png
│   └── ...
│
└── labels\                <-- ALREADY EXISTS (from your script)
    ├── image_01.txt
    └── ...
