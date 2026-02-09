#data Human WHB-10Xv3-Neurons-log2.h5ad 
#source /data/mprc_data1/software/andaconda/bin/activate
#conda activate pytorch_gpu

#mprc15 - Create a clean Python environment
conda create -n cellweb python=3.10 -y
conda activate cellweb
#Install Streamlit 
pip install streamlit numpy pandas scipy matplotlib
#test
streamlit hello

#cd /data/brutus_data34/zhenyao.ye/CrossSpecies/Explore/Human_adata_10x/Supercluster/filterMatch/G20pctS75pct/filterX/subcelltypes/web



#on my own laptop - Create environment
cd C:\Users\mdtsw\Documents
mkdir cell_dense_web
cd cell_dense_web
python -m venv cellweb
cellweb\Scripts\activate.bat
echo %PYTHONPATH%
set PYTHONPATH=
python -m ensurepip --upgrade
python -m pip install --upgrade pip
pip --version
pip install streamlit numpy pandas scipy 
python -m pip install matplotlib seaborn
python -c "import numpy as np; print(np.__version__); import numpy.core._multiarray_umath; print('NumPy OK')"
python -m pip install --upgrade streamlit click
streamlit hello




