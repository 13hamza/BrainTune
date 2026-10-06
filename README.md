# BrainTune

Brain MRI abnormality (tumor) segmentation using a U-Net model, trained on the
[LGG MRI Segmentation dataset](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation).

## Contents
- `trainmodel.ipynb`, `braintumorunet.ipynb`, `brain-abnormality-segmentation-u-net.ipynb`: model training and experiments
- `FYP Beta/`: beta notebook and Flask web app (`mlask/`)
- `myGUI/gui/`: Streamlit GUI app (`app.py`) with PDF report generation

## Running the GUI
```bash
cd myGUI/gui
pip install -r requirements.txt
streamlit run app.py
```
The trained model weights (`unet_brain_mri_seg.h5`) are not included in this repo because they are too large.
