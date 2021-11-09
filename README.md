# HCMUS at Mediaeval21-SportClassificationTask
## Introduction
This is the official repository for our best run in [Mediaeval Challenge-Sport Classification Task](https://multimediaeval.github.io/editions/2021/tasks/sportsvideo/).
Our solution for the problem consists of two stage: individual classification for each component of raw labels, and conditional probability model for producing final results.
Here is the general pipeline of our method.

The method was ranked 2nd place in the leaderboard of the challenge.

## Repository Usage

The mmaction modules of this repository were mainly cloned from [another authors](https://github.com/itruonghai/mmaction2) of our organizations (the original mmaction modules can be
found at [Open-mmlab mmaction2](https://github.com/open-mmlab/mmaction2) )
. We customized the processing flow to adapt with our methods inside three main following folders:
```
${mmaction}
├── first_component
├── second_component
├── third_component
```
### Environment installation

First create conda environment for the project
```
conda create --name mmaction --file customized_requirements.txt
conda activate mmaction
```

### Data and pretrained weights

#### Data

Since the data for Mediaeval21 SportClassification Task was private, readers are suggested to contact [Challenge owners](https://multimediaeval.github.io/editions/2021/tasks/sportsvideo/)
for downloading the data. After that, a sequence of data processing steps was applied before input to our method.

+ Prepare videos:

For the ease of usage, instances of videos corresponding to different frame intervals were extracted with the following script:

```
python utils/prepare_video.py
```

+ Prepare frames:

Then, frames of video can be processed using the following script:
```
python mmaction/tools/data/build_rawframes.py $path_to_extracted_folder --task rgb \
    --ext mp4 --use-opencv
```

+ Prepare annotations:

Finally, annotations for each video instances were extracted for different components. The readers are suggested to change the following code snippets
inside ```utils/prepare_data.py``` for different component options:

```
# Switch between options for prepare independent data

# label_id1 = name_component_to_id_1.index(label[0])
# instance = " ".join([name_instance, str(total_fr), str(label_id1)])

label_id2 = name_component_to_id_2.index(label[1])
instance = " ".join([name_instance, str(total_fr), str(label_id2)])

# label_id3 = name_component_to_id_3.index(label[2])
# instance = " ".join([name_instance, str(total_fr), str(label_id3)])
```

#### Pretrained Weights
Each classifier utilzed pretrained weights which have been trained in kinetics dataset. Download the following weights for intializing each classifier model:
```
mkdir mmaction/checkpoints
wget -c https://download.openmmlab.com/mmaction/recognition/csn/ircsn_ig65m_pretrained_r152_32x2x1_58e_kinetics400_rgb/ircsn_ig65m_pretrained_r152_32x2x1_58e_kinetics400_rgb_20200803-fc66ce8d.pth \
      -O mmaction/checkpoints/ircsn_ig65m_pretrained_r152_32x2x1_58e_kinetics400_rgb_20200803-fc66ce8d.pth
```

### Training

The training process can be divided into three independent training steps: first,second, and third component. Each classifier represents for the learning model
of each component in the raw labels. The table in section **Evaluating** summarized labels of each component in three training stages:

Different training scripts for each component are specified in 
```
# First
python mmaction/first_component/custom_train.py
# Second
python mmaction/second_component/custom_train.py
# Third
python mmaction/third_component/custom_train.py
```

### Evaluating
For each classifier, list of probability scores were produced. We utilized all three lists with the help of our proposed conditional probability models with prior knowledge
to combine them together as final result. For more information about the methods, please refer to our [Working Notes]().

To reproduce our results on validating set, download our best checkpoints for three models at [Best Checkpoint](https://drive.google.com/drive/folders/1bnOK-6rRGch-nic4Qy92OAbsPGmQFzIi?usp=sharing).
Then, use ```custom_val.py`` in each of component folder for performing validation on the data. Here are the accuracy for each components on validation data:

| Component               | Labels | Accuracy |
|--------------------|------|------|
| First Component     | Serve, Offensive, Defensive | 0.9072164948453608 |
| Second Component    | Forehand, Backhand | 0.9690721649484536 |
| Third Component    | Backspin, Loop, Sidespin, Topspin, Hit, Flip, Push, Block| 0.7835051546391752 |

Each of component validation file produces a json result file which will be used in ```combined_triplet_classifier_val.py``` to extract the final labels for the task.
Our method achieves accuracy of **0.7680412371134021** in the validation set



### Testing

To perform inference, the script ```custom_infer.py``` in each component folder was used. Please re-specify the input, output path and checkpoint for running the script. After that,
run the script ```combined_triplet_classifier_test.py```.


### Citation
```
@inproceedings{DBLP:conf/mediaeval/Nguyen-TruongCN20,
  author    = {Hai Nguyen{-}Truong and
               San Cao and
               N. A. Khoa Nguyen and
               Bang{-}Dang Pham and
               Hieu Dao and
               Minh{-}Quan Le and
               Hoang{-}Phuc Nguyen{-}Dinh and
               Hai{-}Dang Nguyen and
               Minh{-}Triet Tran},
  editor    = {Steven Hicks and
               Debesh Jha and
               Konstantin Pogorelov and
               Alba Garc{\'{\i}}a Seco de Herrera and
               Dmitry Bogdanov and
               Pierre{-}Etienne Martin and
               Stelios Andreadis and
               Minh{-}Son Dao and
               Zhuoran Liu and
               Jos{\'{e}} Vargas Quiros and
               Benjamin Kille and
               Martha A. Larson},
  title     = {{HCMUS} at MediaEval 2020: Ensembles of Temporal Deep Neural Networks
               for Table Tennis Strokes Classification Task},
  booktitle = {Working Notes Proceedings of the MediaEval 2020 Workshop, Online,
               14-15 December 2020},
  series    = {{CEUR} Workshop Proceedings},
  volume    = {2882},
  publisher = {CEUR-WS.org},
  year      = {2020},
  url       = {http://ceur-ws.org/Vol-2882/paper50.pdf},
  timestamp = {Mon, 21 Jun 2021 16:26:35 +0200},
  biburl    = {https://dblp.org/rec/conf/mediaeval/Nguyen-TruongCN20.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}

@misc{2020mmaction2,
    title={OpenMMLab's Next Generation Video Understanding Toolbox and Benchmark},
    author={MMAction2 Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmaction2}},
    year={2020}
}
```

### Contact Information

If you have any concerns about this project, please contact:

+ Nguyen Trong Tung(nguyentrongtung11101999@gmail.com)

+ Nguyen Truong Hai(nthai18@apcs.fitus.edu.vn)

