
# DCASE 2023: Sound Event Localization and Detection Evaluated in Real Spatial Sound Scenes

[Please visit the official webpage of the DCASE 2023 Challenge for details missing in this repo](https://dcase.community/challenge2023/task-sound-event-localization-and-detection-evaluated-in-real-spatial-sound-scenes).
   
As the baseline method for the [SELD task](https://www.aane.in/research/computational-audio-scene-analysis-casa/sound-event-localization-detection-and-tracking), we use the SELDnet method studied in the following papers, with  Multiple Activity-Coupled Cartesian Direction of Arrival (Multi-ACCDOA) representation as the output format. Specifically for the microphone version of the dataset, we have added support of the SALSA-lite features. If you are using this baseline method or the datasets in any format, then please consider citing the following papers. If you want to read more about [generic approaches to SELD then check here](https://www.aane.in/research/computational-audio-scene-analysis-casa/sound-event-localization-detection-and-tracking).


1. [Sharath Adavanne, Archontis Politis, Joonas Nikunen and Tuomas Virtanen, "Sound event localization and detection of overlapping sources using convolutional recurrent neural network" in IEEE Journal of Selected Topics in Signal Processing (JSTSP 2018)](https://arxiv.org/pdf/1807.00129.pdf)
2. [Kazuki Shimada, Yuichiro Koyama, Shusuke Takahashi, Naoya Takahashi, Emiru Tsunoo, and Yuki Mitsufuji, " Multi-ACCDOA: localizing and detecting overlapping sounds from the same class with auxiliary duplicating permutation invariant training" in the The international Conference on Acoustics, Speech, & Signal Processing (ICASSP 2022)](https://arxiv.org/pdf/2110.07124.pdf)
3. [Thi Ngoc Tho Nguyen, Douglas L. Jones, Karn N. Watcharasupat, Huy Phan, and Woon-Seng Gan, "SALSA-Lite: A fast and effective feature for polyphonic sound event localization and detection with microphone arrays" in the International Conference on Acoustics, Speech, & Signal Processing (ICASSP 2022)](https://arxiv.org/pdf/2111.08192.pdf)
4. [Parthasaarathy Sudarsanam, Archontis Politis, Konstantinos Drossos, "Assessment of Self-Attention on Learned Features For Sound Event Localization and Detection" in proceedings of the 6th Detection and Classification of Acoustic Scenes and Events 2021 Workshop (DCASE2021)](https://dcase.community/documents/workshop2021/proceedings/DCASE2021Workshop_Sudarsanam_38.pdf)

## BASELINE METHOD

In comparison to the SELDnet studied in [1], we have introduced multi-head self attention blocks [4] in the SELDnet architecture. We have also changed the output format to Multi-ACCDOA [2] to support detection of multiple instances of the same class overlapping. Additionally, we use SALSA-lite [3] features for the microphone version of the dataset, this is to overcome the poor performance of GCC features in the presence of multiple overlapping sound events. 

The final SELDnet architecture is as shown below. The input is the multichannel audio, from which the different acoustic features are extracted based on the input format of the audio. Based on the chosen dataset (FOA or MIC), the baseline method takes a sequence of consecutive feature-frames and predicts all the active sound event classes for each of the input frame along with their respective spatial location, producing the temporal activity and DOA trajectory for each sound event class. In particular, a convolutional recurrent neural network (CRNN), followed by multi-head self attention (MHSA) blocks are used to map the frame sequence to a Multi-ACCDOA sequence output which encodes both [sound event detection (SED)](https://www.aane.in/research/computational-audio-scene-analysis-casa/sound-event-detection) and [direction of arrival (DOA)](https://www.aane.in/research/computational-audio-scene-analysis-casa/sound-event-localization-and-tracking) estimates in the continuous 3D space as a multi-output regression task. Each sound event class in the Multi-ACCDOA output is represented by three regressors that estimate the Cartesian coordinates x, y and z axes of the DOA around the microphone. If the vector length represented by x, y and z coordinates are greater than 0.5, the sound event is considered to be active, and the corresponding x, y, and z values are considered as its predicted DOA.

<p align="center">
   <img src="https://github.com/sharathadavanne/seld-dcase2023/blob/main/images/DCASE2023_seld_baseline.png" width="400" title="SELDnet+Multi-ACCDOA Architecture">
</p>


The figure below visualizes the SELDnet input and outputs for one of the recordings in the dataset. The horizontal-axis of all sub-plots for a given dataset represents the same time frames, the vertical-axis for spectrogram sub-plot represents the frequency bins, vertical-axis for SED reference and prediction sub-plots represents the unique sound event class identifier, and for the DOA reference and prediction sub-plots, it represents the distances along the Cartesian axes. The figures represents each sound event class and its associated DOA outputs with a unique color. Similar plot can be visualized on your results using the [provided script](visualize_seldnet_output.py).

<p align="center">
   <img src="https://github.com/sharathadavanne/seld-dcase2022/blob/main/images/SELDnet_output.jpg" width="300" title="SELDnet input and output visualization">
</p>

## DATASETS

Similar to previous editions of the challenge, the participants can choose either or both the versions or the datasets,

 * **Sony-TAu Realistic Spatial Soundscapes 2023 (STARSS23) - Ambisonic**
 * **Sony-TAu Realistic Spatial Soundscapes 2023 (STARSS23) - Microphone Array**

These datasets contain recordings from an identical scene, with **Ambisonic** version providing four-channel First-Order Ambisonic (FOA) recordings while  **Microphone Array** version provides four-channel directional microphone recordings from a tetrahedral array configuration. Both the datasets, consists of a development and evaluation set. All participants are expected to use the fixed splits provided in the baseline method for reporting the development scores. The evaluation set will be released at a later point.

More details on the recording procedure and dataset can be read on the [DCASE 2023 task webpage](https://dcase.community/challenge2023/task-sound-event-localization-and-detection-evaluated-in-real-spatial-sound-scenes#Dataset).

The development dataset can be downloaded from the link - [**Sony-TAu Realistic Spatial Soundscapes 2023 (STARSS23)**](https://doi.org/10.5281/zenodo.7709052) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7709052.svg)](https://doi.org/10.5281/zenodo.7709052)

The baseline method presented here is trained with a combination of the development set recordings and synthetic recordings. More details about the synthetic dataset used here can be found in the [DCASE 2022 task webpage](https://dcase.community/challenge2022/task-sound-event-localization-and-detection-evaluated-in-real-spatial-sound-scenes#example-external-data-use-with-baseline) 


## Getting Started

This repository consists of multiple Python scripts forming one big architecture used to train the SELDnet.
* The `batch_feature_extraction.py` is a standalone wrapper script, that extracts the features, labels, and normalizes the training and test split features for a given dataset. Make sure you update the location of the downloaded datasets before.
* The `parameter.py` script consists of all the training, model, and feature parameters. If a user has to change some parameters, they have to create a sub-task with unique id here. Check code for examples.
* The `cls_feature_class.py` script has routines for labels creation, features extraction and normalization.
* The `cls_data_generator.py` script provides feature + label data in generator mode for training.
* The `seldnet_model.py` script implements the SELDnet architecture.
* The `SELD_evaluation_metrics.py` script implements the metrics for joint evaluation of detection and localization.
* The `train_seldnet.py` is a wrapper script that trains the SELDnet. The training stops when the SELD error (check paper) stops improving.
* The `cls_compute_seld_results.py` script computes the metrics results on your DCASE output format files. You can switch between using polar or Cartesian based scoring. Ideally both should give identical results.

Additionally, we also provide supporting scripts that help analyse the results.
 * `visualize_seldnet_output.py` script to visualize the SELDnet output


### Prerequisites

The provided codebase has been tested on python 3.8.11 and torch 1.10.0


### Training the SELDnet

In order to quickly train SELDnet follow the steps below.

* For the chosen dataset (Ambisonic or Microphone), download the respective zip file. This contains both the audio files and the respective metadata. Unzip the files under the same 'base_folder/', ie, if you are Ambisonic dataset, then the 'base_folder/' should have two folders - 'foa_dev/' and 'metadata_dev/' after unzipping.

* Now update the respective dataset name and its path in `parameter.py` script. For the above example, you will change `dataset='foa'` and `dataset_dir='base_folder/'`. Also provide a directory path `feat_label_dir` in the same `parameter.py` script where all the features and labels will be dumped.

* Extract features from the downloaded dataset by running the `batch_feature_extraction.py` script. Run the script as shown below. This will dump the normalized features and labels in the `feat_label_dir` folder. The python script allows you to compute all possible features and labels. You can control this by editing the `parameter.py` script before running the `batch_feature_extraction.py` script.

```
python3 batch_feature_extraction.py
```
* You can compute the default parameters for the FOA version of the dataset with Multi-ACCDOA labels using the command below. Check the code to see how it is implemented.

```
python3 batch_feature_extraction.py 3
```

* Similarly to compute the default parameters for the MIC version of the dataset with SALSA-lite features and Multi-ACCDOA labels use the command below.

```
python3 batch_feature_extraction.py 7
```

* Finally, you can now train the SELDnet using the default parameters using
```
python3 train_seldnet.py
```

* Additionally, you can add/change parameters by using a unique identifier \<task-id\> in if-else loop as seen in the `parameter.py` script and call them as following. Where \<job-id\> is a unique identifier which is used for output filenames (models, training plots). You can use any number or string for this.
```
python3 train_seldnet.py <task-id> <job-id>
```

* In order to get baseline results on the development set for Microphone array recordings using SALSA-lite features and Multi-ACCDOA labels, you can run the following command
```
python3 train_seldnet.py 7
```

* Similarly, for Microphone array recordings using GCC-PHAT features and Multi-ACCDOA labels, run the following command
```
python3 train_seldnet.py 6
```

* Finally, for Ambisonic format baseline results, run the following command
```
python3 train_seldnet.py 3
```

* By default, the code runs in `quick_test = True` mode. This trains the network for 2 epochs on only 2 mini-batches. Once you get to run the code sucessfully, set `quick_test = False` in `parameter.py` script and train on the entire data.


## Results on development dataset

As the [SELD evaluation metric](https://www.aane.in/research/computational-audio-scene-analysis-casa/sound-event-localization-detection-and-tracking#h.ragsbsp7ujs) we employ the joint localization and detection metrics proposed in [1], with extensions from [2] to support multi-instance scoring of the same class.

1. [Annamaria Mesaros, Sharath Adavanne, Archontis Politis, Toni Heittola, and Tuomas Virtanen, "Joint Measurement of Localization and Detection of Sound Events", IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA 2019)](https://ieeexplore.ieee.org/document/8937220)

2. [Archontis Politis, Annamaria Mesaros, Sharath Adavanne, Toni Heittola, and Tuomas Virtanen, "Overview and Evaluation of Sound Event Localization and Detection in DCASE 2019", IEEE/ACM Transactions on Audio, Speech, and Language Processing (TASLP 2020)](https://arxiv.org/pdf/2009.02792.pdf)

There are in total four metrics that we employ in this challenge.
The first two metrics are more focused on the detection part, also referred as the location-aware detection, corresponding to the error rate (ER<sub>20°</sub>) and F-score (F<sub>20°</sub>) in one-second non-overlapping segments. We consider the prediction to be correct if the prediction and reference class are the same, and the distance between them is below 20&deg;.
The next two metrics are more focused on the localization part, also referred as the class-aware localization, corresponding to the localization error (LE<sub>CD</sub>) in degrees, and a localization Recall (LR<sub>CD</sub>) in one-second non-overlapping segments, where the subscript refers to _classification-dependent_. Unlike the location-aware detection, we do not use any distance threshold, but estimate the distance between the correct prediction and reference.

The key difference in metrics with previous editions of the challenge is that this year we use the macro mode of computation. We first compute the above four metrics for each of the sound class, and then average them to get the final system performance.

The evaluation metric scores for the test split of the development dataset is given below. 

| Dataset | ER<sub>20°</sub> | F<sub>20°</sub> | LE<sub>CD</sub> | LR<sub>CD</sub> |
| ----| --- | --- | --- | --- |
| Ambisonic (FOA + Multi-ACCDOA) | 0.57 | 29.9 % | 21.6&deg; | 47.7 % |
| Microphone Array (MIC-GCC + Multi-ACCDOA) | 0.62 | 27.8 % | 27.0&deg; | 44.3 % |

**Note:** The reported baseline system performance is not exactly reproducible due to varying setups. However, you should be able to obtain very similar results.

## Submission

* Before submission, make sure your SELD results look good by visualizing the results using `visualize_seldnet_output.py` script
* Make sure the file-wise output you are submitting is produced at 100 ms hop length. At this hop length a 60 s audio file has 600 frames.

For more information on the submission file formats [check the website](https://dcase.community/challenge2023/task-sound-event-localization-and-detection-evaluated-in-real-spatial-sound-scenes#submission)

## License

This repo and its contents have the MIT License.
