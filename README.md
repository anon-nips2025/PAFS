# PAFS

PAFS: Privacy-Aware Face Swapping with Traceable Identity Reconstruction

Place paired face images and matching binary face-region masks in `data/{train,val,test}/{images|masks}`, put `netG_batch_800000.pth` (final Stage-1 weights) and `NoiseLayer256.pth` in `pretrained_weights/`, `arcface_checkpoint.tar` in `arcface_model/`, and train in two phases—run `train_Stage1.py` or skip straight to `train_Stage2.py` using the Stage-1 weights.

```
PAFS/
├── arcface_model/              
│	└── arcface_checkpoint.tar
├── checkpoints_Stage1/         
├── checkpoints_Stage2/         
├── data/
│   ├── train/                  # image ↔ mask names must match 1-to-1
│   │   ├── images/             # e.g. img_0.png
│   │   └── masks/              # e.g. img_0.png (binary face-region mask)
│   ├── val/                    
│   │   ├── images/
│   │   └── masks/
│   └── test/                   
│       ├── images/
│       └── masks/
├── dataset/                     
├── models/                      
├── NoiseLayerNet/               
├── pg_modules/                  
├── pretrained_weights/          
│   ├── netG_batch_800000.pth
│   └── NoiseLayer256.pth
├── util/
│── train_Stage1.py          
│── train_Stage2.py          
└── README.md       
```

