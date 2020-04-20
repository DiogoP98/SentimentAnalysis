
echo ". /home/<user>/miniconda3/etc/profile.d/conda.sh" >> ~/.bashrc
source ~/.bashrc
conda activate preprocessing
conda install pandas
conda install numpy
conda install spacy
pip install spacymoji
