

conda env create -f zooniverse_environment.yml

conda actiavte bubblezoo

# Grab the dev version of reproject
git clone https://github.com/astropy/reproject.git
cd python-reprojection
pip install -e .

