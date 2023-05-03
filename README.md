[![Python](https://img.shields.io/badge/python-3.11.0-blue.svg)](https://www.python.org/)
[![MNE](https://img.shields.io/badge/mne-1.3.1-yellowgreen.svg)](https://pypi.org/project/mne/)
[![MOABB](https://img.shields.io/badge/moabb-0.4.6-lightgrey.svg)](https://pypi.org/project/moabb/)
[![torch](https://img.shields.io/badge/torch-2.0.0+cu118-red.svg)](https://pypi.org/project/torch/)
[![Optuna](https://img.shields.io/badge/optuna-3.1.1-green.svg)](https://pypi.org/project/optuna/)
[![torch-summary](https://img.shields.io/badge/torch_summary-1.4.5-orange.svg)](https://pypi.org/project/torch-summary/)
[![scikit-learn](https://img.shields.io/badge/scikit_learn-1.2.2-red.svg)](https://pypi.org/project/scikit-learn/)
# BCI Competition Graz 2a Motor Imagery Classification

This repository contains the implementation of a motor imagery classification model using the BCI Competition Graz Dataset 2a. The model is based on the EEGNet architecture and demonstrates the classification of four motor imagery classes. The dataset description can be found [here](resources/desc_2a.pdf).

## Dataset

The BCI Competition Graz Dataset 2a consists of EEG recordings from subjects performing four different motor imagery tasks (left hand, right hand, both feet, and tongue). The dataset is divided into two sessions (runs) per subject, recorded on different days.

---

## Generators

There are two generators implemented in this repository:

1. Cross Run Generator: This generator is used for training and validation of the model within a single subject, using one run for training and the other run for validation. This allows the assessment of the model's performance in a within-subject scenario.

2. Cross Subject Generator: This generator is used for training and validation of the model across different subjects, using a leave-one-subject-out cross-validation scheme. This allows the assessment of the model's generalizability across different individuals.

You can try both generators using the provided example notebook: [dataset_example.ipynb](./dataset_example.ipynb).

---

## EEGNet Architecture

The model implemented in this repository is based on the EEGNet architecture, a compact and efficient Convolutional Neural Network (CNN) architecture designed specifically for decoding raw EEG signals. For more information about the EEGNet architecture, please refer to the original publication:

- [Lawhern, V. J., Solon, A. J., Waytowich, N. R., Gordon, S. M., Hung, C. P., & Lance, B. J. (2018). EEGNet: a compact convolutional network for EEG-based brain–computer interfaces. Journal of neural engineering, 15(5), 056013.](https://doi.org/10.1088/1741-2552/aace8c)

---

## Example Train Notebook

You can train the model using the provided example notebook: [train.ipynb](./train.ipynb).

---

## Train (Cross Run Validation)

This repository includes a Python script for performing cross run validation training on a single subject. The script takes the subject ID as input and can also accept an optional configuration file placed in the `config/` directory.

### Usage

To run the cross run validation training script, execute the following command in your terminal:

```
|python train_CrossRunValidation.py <subject_id> --config <config_file>
```


Replace `<subject_id>` with the ID of the subject you want to train on, and `<config_file>` with the name of the configuration file (without the file extension) placed in the `config/` directory. If you do not provide a configuration file, the script will use the default configuration.

### Example

To train the model on subject 1 with the default configuration, run the following command:

```
|python train_CrossRunValidation.py 1
```

Note: Make sure the custom configuration file is placed in the `config/` directory.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
