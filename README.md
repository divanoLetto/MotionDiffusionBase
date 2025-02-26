# Motion Diffusion Base 

---

![](https://img.shields.io/github/contributors/divanoLetto/MotionCompositionDiffusion?color=light%20green) ![](https://img.shields.io/github/repo-size/divanoLetto/MotionCompositionDiffusion?cacheSeconds=60)

---

## Installation

1. Create Conda virtual environment:

    ```
    conda create -n MDB python=3.10.9
    conda activate MDB
    ```
   
2. Install PyTorch following the [official istructions](https://pytorch.org/get-started/locally/):
    ```
    conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
    ```

4. Clone this repository and install its requirements:
    ```
    git clone https://github.com/divanoLetto/MCD
    conda install pytorch-lightning
    conda install conda-forge::hydra-core
    conda install conda-forge::colorlog
    conda install conda-forge::orjson
    conda install conda-forge::einops
    pip install smplx
    pip install trimesh
    conda install conda-forge::opencv
    conda install conda-forge::moviepy
    pip install git+https://github.com/openai/CLIP.git
    conda install matplotlib
    pip install pyrender
    conda install -c conda-forge transformers
    conda install conda-forge::openai
    conda install conda-forge::python-dotenv
    ```

# Dataset Setup

This work uses the SMPL+H data format, an extension of the SMPL model that includes articulated hand poses, enabling more detailed human motion representation. The data will be structured as (bs, num_frames, 205 features), where each feature vector encodes body shape, pose parameters, and hand articulation.

#### SMPL dependency

Please follow the following istructions to obtain the ``deps`` folder with SMPL+H downloaded, and place ``deps`` in the base directory.

<details><summary>Download distilbert instructions</summary>

#### Download distilbert from __Hugging Face__
```bash
mkdir deps
cd deps/
git lfs install
git clone https://huggingface.co/distilbert-base-uncased
cd ..
```

</details>

<details><summary> SMPL body model instructions</summary>

This is only useful if you want to use generate 3D human meshes like in the teaser. In this case, you also need a subset of the AMASS dataset (see instructions below).

Go to the [MANO website](https://mano.is.tue.mpg.de/download.php), register and go to the Download tab.

- Click on "Models & Code" to download ``mano_v1_2.zip`` and place it in the folder ``deps/smplh/``.
- Click on "Extended SMPL+H model" to download ``smplh.tar.xz`` and place it in the folder ``deps/smplh/``.

The next step is to extract the archives, merge the hands from ``mano_v1_2`` into the ``Extended SMPL+H models``, and remove any chumpy dependency.
All of this can be done using with the following commands. (I forked both scripts from this repo [SMPLX repo](https://github.com/vchoutas/smplx/tree/master/tools), updated them to Python 3, merged them, and made it compatible with ``.npz`` files).


```bash
pip install scipy chumpy
bash prepare/smplh.sh
```

This will create ``SMPLH_FEMALE.npz``, ``SMPLH_MALE.npz``, ``SMPLH_NEUTRAL.npz`` inside the ``deps/smplh`` folder.

The error `ImportError: cannot import name 'bool' from 'numpy'` may occur depending on the versions of numpy and chumpy, in this case try commenting out the row that throws the exception: `from numpy import bool, int, float, complex, object, unicode, str, nan, inf`.

</details>

#### Motions

<details><summary>Click to expand</summary>

The motions all come from the AMASS dataset. Please download all "SMPL-H G" motions from the [AMASS website](https://amass.is.tue.mpg.de/download.php) and place them in the folder ``datasets/motions/AMASS``.

<details><summary>It should look like this:</summary>

```bash
datasets/motions/
└── AMASS
    ├── ACCAD
    ├── BioMotionLab_NTroje
    ├── BMLhandball
    ├── BMLmovi
    ├── CMU
    ├── DanceDB
    ├── DFaust_67
    ├── EKUT
    ├── Eyes_Japan_Dataset
    ├── HumanEva
    ├── KIT
    ├── MPI_HDM05
    ├── MPI_Limits
    ├── MPI_mosh
    ├── SFU
    ├── SSM_synced
    ├── TCD_handMocap
    ├── TotalCapture
    └── Transitions_mocap
```

Each file contains a "poses" field with 156 (52x3) parameters (1x3 for global orientation, 21x3 for the whole body, 15x3 for the right hand and 15x3 for the left hand).

</details>

Then, launch these commands:

```bash
python prepare/amasstools/fix_fps.py
python prepare/amasstools/smpl_mirroring.py
python prepare/amasstools/extract_joints.py
python prepare/amasstools/get_smplrifke.py
```

<details><summary>Click here for more information on these commands</summary>

#### Fix FPS

The script will interpolate the SMPL pose parameters and translation to obtain a constant FPS (=20.0). It will also remove the hand pose parameters, as they are not captured for most AMASS sequences. The SMPL pose parameters now have 66 (22x3) parameters (1x3 for global orientation and 21x3 for full body). It will create and save all the files in the folder ``datasets/motions/AMASS_20.0_fps_nh``.


#### SMPL mirroring

This command will mirror SMPL pose parameters and translations, to enable data augmentation with SMPL (as done by the authors of HumanML3D with joint positions).
The mirrored motions will be saved in ``datasets/motions/AMASS_20.0_fps_nh/M`` and will have a structure similar than the enclosing folder.


#### Extract joints

The script extracts the joint positions from the SMPL pose parameters with the SMPL layer (24x3=72 parameters). It will save the joints in .npy format in this folder: ``datasets/motions/AMASS_20.0_fps_nh_smpljoints_neutral_nobetas``.


#### Get SMPL RIFKE

This command will use the joints + SMPL pose parameters (in 6D format) to create a unified representation (205 features). Please see ``prepare/amasstools/smplrifke_feats.py`` for more details.

</details>

The dataset folder should look like this:
```bash
datasets/motions
├── AMASS
├── AMASS_20.0_fps_nh
├── AMASS_20.0_fps_nh_smpljoints_neutral_nobetas
└── AMASS_20.0_fps_nh_smplrifke
```

</details>

#### Text

<details><summary>Click to expand</summary>

Next, run the following command to pre-compute the CLIP embeddings (ViT-B/32):

```
python -m prepare.embeddings
```

The folder should look like this:
```
datasets/annotations/humanml3d
├── annotations.json
├── splits
│   ├── all.txt
│   ├── test_tiny.txt
│   ├── test.txt
│   ├── train_tiny.txt
│   ├── train.txt
│   ├── val_tiny.txt
│   └── val.txt
└── text_embeddings
    └── ViT-B
        ├── 32_index.json
        ├── 32.npy
        └── 32_slice.npy
```

</details>

## Download pretrained models

Click on this [link](https://drive.google.com/file/d/1s8yHZQwO0rDK_a3JeCGDSWrU-DgQuKE7) and download from your web browser.
Then extract in `pretrained_models/` dir.

## Usage

### Train a model 
Start the training process with:
```
python train.py trainer.max_epochs=10000 +split='complex/train' run_dir='outputs/mdm-smpl_splitcomplex_humanml3d/logs/checkpoints/'
```
It will save the outputs to the folder specified by the argument `run_dir`. The number of diffusion steps is defined by the argument `trainer.max_epochs`. \
The default dataset is HumanML3D, it can be changed to KitML using the argument `dataset=kitml`. To resume a training it's possible to use the argument `resume_dir` and `ckpt`, where the first specify the base directory where the config of the model are saved and the latter specify the path of the *.ckpt* file.

### Generate 

```
python generate.py input=\'inputs/example.txt\' ckpt='logs/checkpoints/last.ckpt' run_dir='pretrained_models/mdm-smpl_clip_smplrifke_humanml3d' out_formats=\[\'txt\',\'videojoints\',\ \'videosmpl\',\ \'joints\'\]
```

- `run_dir` the path of the directory where to find the configs of the models and where to save results
- `ckpt` the path of the .ckpt file inside of the `run_dir` 
- `out_formats` output formats to be saved.