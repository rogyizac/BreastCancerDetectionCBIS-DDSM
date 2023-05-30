# Breast Cancer Detection CBIS-DDSM
A VGG16 tumor classification model built on the [CBIS DDSM](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=22516629) Dataset. Here we use the model to classify benign calcifications from malignant calcifications

See the main.ipynb for validation results and metrics.
Confusion Matrix:
| 175 | 33 |
|-----|----|
| 31  | 70 |
valid Loss: 0.2169 Acc: 0.7929

To run the code,
- Download the data
- Change paths to the training and metadata file in the main.py/ipynb file. (metadata file is downloaded along with the data download)
- Run the main.py/ipynb file
