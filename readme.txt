### readme.txt ###
The datasets of images and annotation files are necessary under the following directory as assgined names.
using_dataset/annotation
using_dataset/dataset_img

The following extracted feature vectors are necessary under the following directory as assigned names and as .txt file.
using_dataset/dataset_nlp
using_dataset/dataset_headpose



+Procedure
1:run all_under1.sh.
  Docker build stard and run bult docker envornment.
2:In docker, run top_all.sh.
  This top_all.sh run the followings.
  Data spliting, extracting image features, pca, concatenating features, classifier, and submission making files.
