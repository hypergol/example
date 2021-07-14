python3 -m venv .venv
source .venv/bin/activate
pip3 install --upgrade pip
pip3 install setuptools==57.1.0
pip3 install wheel
pip3 install -r requirements.txt
# setup here
python3 -m spacy download en_core_web_sm
