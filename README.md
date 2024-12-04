## Deepfake Audio Detection

### Setup
Install the required dependencies. Download the GPU version of these package if required.
``` bash
pip install -r requirements.txt
```
Download and unzip AsvSpoof 2019 LA files

```bash
!wget https://datashare.ed.ac.uk/bitstream/handle/10283/3336/LA.zip
!unzip LA.zip
```

This should give you the following dir:
``` bash
audio-deepfake
├── LA
│   ├── ASVspoof2019_LA_asv_protocols
│   ├── ASVspoof2019_LA_asv_scores
│   ├── ASVspoof2019_LA_cm_protocols
│   ├── ASVspoof2019_LA_dev
│   │   └── flac
│   ├── ASVspoof2019_LA_eval
│   │   └── flac
│   └── ASVspoof2019_LA_train
│       └── flac
└── src
    ├── feature.py
    ├── metrics.py
    ├── model
    ├── protocol
    ├── requirements.txt
    └── run.py

```
Run the experiment by the following command:

``` bash
python run.py \
    -m 'lcnn' \
    -f 'cqt' \
    --lr 0.00001 \
    --epochs 100 \
    --batch 32 \
```
Before you run the experiement on the full dataset, you can set the dataset size to 1000 and verbose to 1 for quick verification.
```bash
python run.py \
    -m 'lcnn' \
    -f 'cqt' \
    --lr 0.00001 \
    --epochs 100 \
    --batch 32 \
    --datasize 1000 \
    --verbose 1 \
    --savedata False
```

