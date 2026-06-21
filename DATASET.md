## Dataset Preparation

Download the STAR dataset [images](https://huggingface.co/datasets/Zhuzi24/STAR) and [annotation files](https://huggingface.co/datasets/Zhuzi24/STAR), then unzip them.

The downloaded images are separated into TRAIN, VAL, and TEST splits. Merge all images into one directory named `STAR_img`.

Place the images and annotation files under `STAR_SGG` with the following structure:

```text
STAR_SGG/
├── STAR_img     # Contains all images, total 1273.
├── STAR-SGG-with-attri.h5
├── ...
├── ...
└── STAR-SGG-dicts-with-attri.json
```

Set the dataset root in `maskrcnn_benchmark/config/paths_catalog.py` by updating `DatasetCatalog.DATA_DIR`:

```python
DATA_DIR = "/path/to/SGG-ToolKit/data/STAR_SGG/"
```

If you use the provided SARNet scripts, also make sure the GloVe files are available under:

```text
data/STAR_SGG/glove
```
