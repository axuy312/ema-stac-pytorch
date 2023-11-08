#!/bin/bash
timestamp() {
  date +"%T" # current time
}

echo "Start script"
timestamp

nvidia-smi

conda -V

#conda env stac
if conda env list | grep "stac" >/dev/null 2>&1; then
    echo "exist"
    conda remove --name stac --all -y
fi
conda create -n stac python=3.6 -y
eval "$(conda shell.bash hook)"
conda activate stac

#ENV Info
echo "===========ENV Info============"
python -V
pip -V

echo "Env setup finish"
timestamp

#setup projeect
echo "===========setup project============"
git clone https://github.com/axuy312/stac-pytorch.git
cd stac-pytorch
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
mkdir logs
mkdir dataset
cd dataset
mkdir PCB
cd PCB
mkdir Annotations_pseudo_label
#Download
echo "===========Downloading dataset============="
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=FILEID' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1GWzz1omosp4GZ_jN60vLIdXuZbggJNDl" -O pcb_dataset.zip && rm -rf /tmp/cookies.txt
unzip pcb_dataset.zip
ls
cd ../../model_data
echo "===========Downloading pretrain weight============"
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=FILEID' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1zHvGrjsodLTUWJncKZsLp128jaatEo42" -O pcb_weights_resnet.pth && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=FILEID' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1DM6okPI1MELM_ktj6FzUYGy-XOiL7SDN" -O voc_weights_vgg.pth && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=FILEID' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1YqN2M5YVKpFc1T4oyWg1gR0gBC4VN5pr" -O voc_weights_resnet.pth && rm -rf /tmp/cookies.txt
cd ..

echo "download finish"
timestamp

python xml2txt.py
python train.py
echo "training finish"
timestamp
