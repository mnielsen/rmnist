# Configuration information for running on Google Compute Cloud.
curl -O http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
dpkg -i ./cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
sudo apt-get update
sudo apt-get install cuda-8-0 -y

# Updates
sudo apt-get --assume-yes upgrade
sudo apt-get --assume-yes install tmux build-essential gcc g++ make binutils
sudo apt-get --assume-yes install software-properties-common

# Anaconda
wget "https://repo.continuum.io/archive/Anaconda3-4.3.0-Linux-x86_64.sh" -O "Anaconda3-4.3.0-Linux-x86_64.sh"
bash Anaconda3-4.3.0-Linux-x86_64.sh -b
echo "export PATH=\"$HOME/anaconda3/bin:\$PATH\"" >> ~/.bashrc
export PATH="$HOME/anaconda3/bin:$PATH"
conda upgrade -y --all

# pytorch, ipython, matplotlib
conda install -y pytorch torchvision cuda80 -c soumith

