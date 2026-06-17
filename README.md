data prep for training yolo11s IR detection model on imagery with hot pixels ("spicy")

Folder Structure 
```text
C:\seals_project\          <-- Main Project Folder
├── train.py               <-- Your script
├── validate.py            <-- Your validation script
├── data.yaml              <-- Your config
│
├── images\                <-- Training imagery folder 
│   ├── train_image_01.png
│   └── ...
│
├── labels\                <-- Labels for training imagery 
    ├── train_image_01.txt  
    └── ...
├── validation\            <-- Validation dataset folder
    ├── images\                 
│       ├── val_image_01.png
│       └── ...
│
    ├── labels\                
        ├── val_image_01.txt  
        └── ...
``` 

Trained yolo11s for speed and accuracy
600 epochs, stop after 50 decline
included speckling ("spice") augmentation setting random pixels to max value (255) 

best.pt weights used in final pipeline

Final pipeline (2025_batch_norm_yolo11s.py):
1) reference manifest of imagelists to process
2) normalize imagery (despice [replace hot pixel with ave of surrounding values], 0.001% min/max, linear stretch)
3) run spicy v11s best.pt model on image
4) do not save normalized image
5) save detections to VIAME csv format




