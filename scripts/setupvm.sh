# Setup Ubuntu
sudo apt update --yes
sudo apt upgrade --yes

# Get Miniconda and make it the main Python interpreter
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh -b -p ~/miniconda 
rm ~/miniconda.sh
echo "PATH=\$PATH:\$HOME/miniconda/bin" >> .bash_profile
echo ". /home/<user>/miniconda3/etc/profile.d/conda.sh" >> ~/.bashrc
source ~/.bash_profile
source ~/.bashrc
conda create -n preprocessing python=3.7
conda activate preprocessing
