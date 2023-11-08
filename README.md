ema-stac-pytorch

## 環境
Ubuntu == 20.04  
Python == 3.6  
Pytorch == 1.9.0  
Anaconda


## Setup
	// conda env stac
	conda create -n stac python=3.6 -y
	conda activate stac  
 	
	// setup projeect
 	cd stac-pytorch
	pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
	pip install -r requirements.txt
 	
  	// Download dataset
 	wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=FILEID' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1r8hfKvV62gBjDo22UK5drUCGsmrSfE61" -O pcb_dataset.zip && rm -rf /tmp/cookies.txt
	unzip pcb_dataset.zip

  	// Download pretrain weight
 	cd model_data
	wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=FILEID' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1DM6okPI1MELM_ktj6FzUYGy-XOiL7SDN" -O voc_weights_vgg.pth && rm -rf /tmp/cookies.txt
	wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=FILEID' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1YqN2M5YVKpFc1T4oyWg1gR0gBC4VN5pr" -O voc_weights_resnet.pth && rm -rf /tmp/cookies.txtc
	cd ..

