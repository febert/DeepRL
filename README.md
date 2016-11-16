# DeepRL


# How to create a python working environment:
virtualenv --system-site-packages deeprl-python (not recommented in python 3)
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
If CUDA is not installed, install it to any directory (no root needed) and add it to the path variables.


# How to get remote access to Jupyter notebook and Tensorboard
	* get them launched in the host
		jupyter-notebook --no-browser
		tensorboard --logdir=./wherever/your/data/is
	* tunnel from the client
		# jupyter
		ssh -NL <local_port>:localhost:8888 host01
		# tensorboard
		ssh -NL <other_local_port>:0.0.0.0:6006
	* access through your browser with localhost:<local_port> and localhost:<other_local_port>

# How to renew kerberos credentials in one line
while true; do echo "password" | kinit; while true; do krenew; if [ $? -ne 0 ]; then break; fi; sleep 30m; done; done

# DQN from_pixels
A small modification of OpenAI's gym is required for efficiently obtaining images from the classic environments avoiding on-screen rendering. The changes can be found in https://github.com/garibarba/gym/commit/69c58d91d64cf3b28c44077bc30c599ed354af1e. Then a minimal change must be done in each environment in order to call the newly defined functions instead of the regular one.

# xvfs, GLX and nvidia drivers
These don't play well with each other. A solution is possible but requires reinstalling the nvidia drivers:
https://davidsanwald.github.io/ec2-openAI-gym-tensorflow-GPU-cuda-deep-learning.html#ec2-openAI-gym-tensorflow-GPU-cuda-deep-learning
