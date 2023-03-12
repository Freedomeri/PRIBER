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

## Start
1, You should first finetune a Bert model by specifing task and dataset. You can refer to our colab notebook:"train_PRIBER.ipynb".
Rename the model file to "pytorch_model.bin" and store it together with json file "config" and text file "vocab".

2, Run "Two_parties_infer.py" with hyper-parameters:
--task Structured_Beer --
