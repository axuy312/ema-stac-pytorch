#!/bin/bash
log=envSetupLog.txt

timestamp() {
  date +"%T" # current time
}

echo "Start script"
timestamp

nvidia-smi

sudo apt install git wget -y

if !(command -v conda &> /dev/null)
then
    echo "conda could not be found"
    wget https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh
    bash Anaconda3-2023.09-0-Linux-x86_64.sh
    source ~/.bashrc
    conda config --set auto_activate_base false
    echo "conda install finish"
fi
timestamp
echo "Finish"
