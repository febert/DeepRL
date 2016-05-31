# DeepRL


# How to create a python working environment:
virtualenv --system-site-packages deeprl-python
source ./deeprl-python/bin/activate
pip install --upgrade pip
	# check if there is a newer version. This link is only for 0.8
# CPU
pip install --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.8.0-cp27-none-linux_x86_64.whl
# GPU
pip install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.8.0-cp27-none-linux_x86_64.whl
pip install jupyter
pip install gym[all]


# How to add an RSA-public key to a remote computer:

ssh-copy-id -i id_rsa.pub user@hostname.example.com
# then add to ~/.ssh/config something like
Host host01
	Hostname hostname01.example.com
	User user21


# How to configure CUDA and cuDNN for tensorflow 0.8
1. make sure cuda-7.5 is installed
# if cuDNN is already installed
	2. add to .bashrc
		 export PATH=/usr/local/cuda-7.5/bin:$PATH
		 export LD_LIBRARY_PATH=/usr/local/cuda-7.5/lib64:$LD_LIBRARY_PATH
# else
	2. get a free developer account to download cuDNN and download it
	3. move it to the remote computer and untar
		$ scp cudnn.tgz host01:~/
		$ tar -xvzf cudnn.tgz
	4. add to .bashrc
		 export PATH=/usr/local/cuda-7.5/bin:$PATH
		 export LD_LIBRARY_PATH=~/cuda/lib64:$LD_LIBRARY_PATH


# How to get remote access to Jupyter notebook and Tensorboard
	* get them launched in the host
		jupyter-notebook --no-browser
		tensorboard --logdir=./wherever/your/data/is
	* tunnel locally to them
		# jupyter
		ssh -NL <local_port>:localhost:8888 host01
		# tensorboard
		ssh -NL <other_local_port>:0.0.0.0:6006
	* access through your browser with localhost:<local_port> and localhost:<other_local_port>
