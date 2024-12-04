## Deepfake Audio Detection

### Cloning the Repository and GitHub Commands Introduction

The first thing you'll need to do is clone the repository. You will only need to do this once. In your local terminal, enter the following command:

```bash
git clone https://github.com/tzujohsu/audio-deepfake-detection.git
```

To switch into this newly cloned local repo, use the following command:

```bash
cd audio-deepfake-detection
```

It is good practice before you make any local changes to always retrieve or "pull", any changes that others have made. You can do so with the following command:

```bash
git pull
```

After making local changes and testing out your code, you can send, or "push", your changes.

First, add the files you changed with the following command: 

```bash
git add -u
```

If you *created* any new files instead of or in addition to changing existing files, use the following command:

```bash
git add [filename]
```

or

```bash
git add --all
```

Next, commit your files. You can do so with this command:

```bash
git commit -m "[a brief message describing your changes]"
```

Finally, push your changes with the following command:

```bash
git push origin main
```

If all goes well, your changes will be pushed to the repo where everyone can see and pull them.

### Setup
Install the required dependencies. Download the GPU version of these package if required.
``` bash
pip install -r requirements.txt
```
Download and unzip AsvSpoof 2019 LA files

```bash
wget https://datashare.ed.ac.uk/bitstream/handle/10283/3336/LA.zip
unzip LA.zip
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

Before running the experiment, remember to change the path to include your uniqname:

```
path_to_database = "/home/[uniqname]/audio-deepfake-detection/" + access_type
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

