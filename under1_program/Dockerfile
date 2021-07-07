#####################################################
# Base image : OS is Ubuntu18.04. 
#              python version is 3.6.8
#####################################################
FROM tensorflow/tensorflow:1.14.0-gpu-py3
MAINTAINER tyamamoto <yamamoto.t.0104@fujitsu.com>

#ENV http_proxy "http://mousecp:1234567890@snd.proxy.nic.fujitsu.com:8080"
#ENV https_proxy "http://mousecp:1234567890@snd.proxy.nic.fujitsu.com:8080"

# install application by using apt-get #
RUN apt-get update && apt-get install -y \
git \
vim \
wget \
graphviz
##########################################

### python library, version assinged ###
RUN pip install keras==2.3.0 \
graphviz==0.11 \
keras-applications==1.0.8 \
keras-preprocessing==1.1.0 \
munkres==1.0.12 \
pydot==1.2.4 \
pydotplus==2.0.2 \
tqdm==4.28.1 \
numpy==1.17.2
#######################

### python library, version not assinged ###
RUN pip install pillow \
matplotlib \
scikit-image \
scikit-learn \
pandas \
catboost \
lightgbm \
seaborn
##############

# open cv install #
RUN pip install opencv-contrib-python
###################

VOLUME /work
WORKDIR /work

# making necessary directory #
RUN mkdir /root/.keras/ && \ 
mkdir /home/ubuntu/ && \
mkdir /home/ubuntu/iccv2021_abaw/ && \
mkdir /home/ubuntu/iccv2021_abaw/abaw_dataset/ && \
mkdir /root/docker_common_directory/
#############################

# copy run directory #
#COPY ./cvpr2020_resnet101_final_rev_2019_1219/ /work/
#############################

#CMD pip list | less > pip_list.txt
