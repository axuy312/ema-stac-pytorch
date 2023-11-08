![image](https://github.com/axuy312/ema-stac-pytorch/assets/44252923/40750584-6b65-47bb-9c92-5d219e4a9fe8)# ema-stac-pytorch

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

 
