# Multi-modal Affect Analysis using standardized data within subjects in the Wild

Challenges: ICCV 2021: 2nd Workshop and Competition on Affective Behavior Analysis in-the-wild (ABAW)

URL: https://ibug.doc.ic.ac.uk/resources/iccv-2021-2nd-abaw/

Team Name: FLAB2021

Team Members: Sachihiro Youoku, Junya Saito, Takahisa Yamamoto, Akiyoshi Uchida, Xiaoyu Mi (1), Ziqiang Shi, Liu Liu, Zhongling Liu (2)

Affiliation (1): Advanced Converging Technologies Laboratories, Fujitsu Ltd., Japan

Affiliation (2): Fujitsu R&D Center Co. Ltd., China

The paper link: Multi-modal Affect Analysis using standardized data within subjects in the Wild

# Necessary data
The datasets of images and annotation files are necessary under the following directory as assgined names.
using_dataset/annotation
using_dataset/dataset_img

The following extracted feature vectors are necessary under the following directory as assigned names and as .txt file.
using_dataset/dataset_nlp
using_dataset/dataset_headpose



# Procedure
1:run all_under1.sh.
  Docker build stard and run bult docker envornment.
2:In docker, run top_all.sh.
  This top_all.sh run the followings.
  Data spliting, extracting image features, pca, concatenating features, classifier, and submission making files.

# Copyright
Copyright 2021 FUJITSU LIMITED.
