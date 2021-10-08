# Physionet for Python and Deep learning
This code will load and convert physionet edf file to binary file in NumPy. Also, an example of how to use these binary files as a Generator for TensorFlow is included

# Getting Started

First, clone this source code. Then, download the dataset "EEG Motor Movement/Imagery Dataset" of the Physionet datasets IV-2a. Put all files of the dataset into a folder.
link to the dataset: https://physionet.org/content/eegmmidb/

# Prerequisites
- Python == 3.7 or 3.8
- tensorFlow == 2.X 
- numpy
- mne
- pyedflib
- sklearn
```bash
pip install -r requirements.txt
```

# Run
```bash
python main.py --help
Physionet Data Creator
optional arguments:
  -h, --help            show this help message and exit
  --path PATH           path to the dataset folder
  --output OUTPUT       path to the converted data
  --valid_test_split    whether to split train data into validation and test

python main.py --path '/physionet.org/files/eegmmidb/1.0.0/' --output 'output' --valid_test_split
```
# Tensorflow generator
see the ```generator_exmaple.py``` file to learn how to use these numpy binaries as a feed for training a TensorFlow model 


# Authors
- Javad Sameri        sameryq@gmail.com
