# DeepRL

# How to create a python working environment:
virtualenv --system-site-packages deeprl-python
source ./deeprl-python/bin/activate
pip install --upgrade pip
	# check if there is a newer version. This link is only for 0.8
pip install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.8.0-cp27-none-linux_x86_64.whl
pip install jupyter
pip install gym[all]
