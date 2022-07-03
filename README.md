# This is a test project of applying SMPC method secret sharing on BERT model


## Requirements

* Python 3.7.5
* PyTorch >= 1.8.1, <=1.10
* Spacy with the ``en_core_web_lg`` models
* NVIDIA Apex (fp16 training)

Install required packages
```
conda install -c conda-forge nvidia-apex
pip install -r requirements.txt
python -m spacy download en_core_web_lg
```
