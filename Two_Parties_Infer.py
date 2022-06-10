from apex import amp
import torch
import numpy as np
import crypten
from ditto.exceptions import ModelNotFoundError
crypten.init()
import json
import jsonlines
import sys
import time
from tqdm import tqdm
sys.path.insert(0, "Snippext_public")
from torch.utils import data
from snippext.model import MultiTaskNet
from ditto.exceptions import ModelNotFoundError
from ditto.dataset import DittoDataset
from ditto.summarize import Summarizer
from ditto.knowledge import *
import os
import MyBertModel


def to_str(row, summarizer=None, max_len=256, dk_injector=None):
    """Serialize a data entry

    Args:
        row (Dictionary): the data entry
        summarizer (Summarizer, optional): the summarization module
        max_len (int, optional): the max sequence length
        dk_injector (DKInjector, optional): the domain-knowledge injector

    Returns:
        string: the serialized version
    """
    # if the entry is already serialized
    if isinstance(row, str):
        return row
    content = ''
    for attr in row.keys():
        content += 'COL %s VAL %s ' % (attr, row[attr])

    if summarizer is not None:
        content = summarizer.transform(content, max_len=max_len)

    if dk_injector is not None:
        content = dk_injector.transform(content)

    return content


def load_model(task, saved_state, lm, use_gpu, fp16=False,path = None):
    """Load a model for a specific task.

    Args:
        task (str): the task name
        path (str): the path of the checkpoint directory
        lm (str): the language model
        use_gpu (boolean): whether to use gpu
        fp16 (boolean, optional): whether to use fp16

    Returns:
        Dictionary: the task config
        MultiTaskNet: the model
    """
    # load models

    configs = json.load(open('configs.json'))
    configs = {conf['name'] : conf for conf in configs}

    if use_gpu:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = 'cpu'

    config = configs[task]
    config_list = [config]
    model = MultiTaskNet([config], device, False, lm=lm,bert_path = path)

    model.load_state_dict(saved_state)


    model = model.to(device)

    if fp16 and 'cuda' in device:
        model = amp.initialize(model, opt_level='O2')

    return config, model



def load_dataset(input_path,config,max_len = 128,summarizer=None,dk_injector=None):
    # input_path can also be train/valid/test.txt
    # convert to jsonlines
    if '.txt' in input_path:
        with jsonlines.open(input_path + '.jsonl', mode='w') as writer:
            for line in open(input_path):
                writer.write(line.split('\t')[:2])
        input_path += '.jsonl'

    # batch processing
    start_time = time.time()
    with jsonlines.open(input_path) as reader:
        pairs = []
        rows = []
        for idx, row in tqdm(enumerate(reader)):
            pairs.append((to_str(row[0], summarizer, max_len, dk_injector),
                          to_str(row[1], summarizer, max_len, dk_injector)))
            rows.append(row)

    run_time = time.time() - start_time
    run_tag = '%s_lm=%s_dk=%s_su=%s' % (config['name'], lm, str(dk_injector != None), str(summarizer != None))
    os.system('echo %s %f >> log.txt' % (run_tag, run_time))
    return pairs,rows

if __name__ == "__main__":

    ALICE = 0
    BOB = 1

    '''static configs'''
    use_gpu = False
    lm = 'bert-small'
    task = 'Structured_Beer'
    max_len = 128
    model_path = "/home/smz/models/bert-small-finetuned"

    '''Split saved Parameters into embeddings and main model params'''
    BertEmbeds, saved_state = MyBertModel.SplitModel(model_path)

    '''load server model'''
    taskConfig,serverModel = load_model(task,saved_state,lm,use_gpu,path = model_path)

    '''load client dataset'''
    dataPairs,dataRows = load_dataset('input/input_beer.txt',taskConfig,max_len=max_len)
    inputs = []
    for (sentA, sentB) in dataPairs:
        inputs.append(sentA + '\t' + sentB)

    dataset = DittoDataset(inputs, taskConfig['vocab'], taskConfig['name'], lm=lm, max_len=max_len)
    iterator = data.DataLoader(dataset=dataset,
                               batch_size=128,
                               shuffle=False,
                               num_workers=0,
                               collate_fn=DittoDataset.pad)
    input_embeddings = []
    Y = []
    with torch.no_grad():
        # print('Classification')
        for i, batch in enumerate(iterator):
            words, x, is_heads, tags, mask, y, seqlens, taskname = batch
            taskname = taskname[0]
            input_embeddings.append(MyBertModel._BertEmbedding(x, BertEmbeds,hidden_size=512))
            Y.append(y)

    print(len(input_embeddings))
    '''divide into 2 parts(for test)'''
    temp1 = input_embeddings[0][:64]
    temp2 = input_embeddings[0][64:]
    yTemp1 = Y[0][:64]
    yTemp2 = Y[0][64:]
    input_embeddings[0] = temp1
    input_embeddings.append(temp2)

    '''encrypt and process for 2 parties'''
    from crypten import mpc
    import crypten.communicator as comm
    torch.set_num_threads(1)

    @mpc.run_multiprocess(world_size=2)
    def saveData(input_embeddings):
        samples_alice = input_embeddings[0]
        samples_bob = input_embeddings[1]
        crypten.save_from_party(samples_alice, "encrypted_data/samples_alice.pt", src=ALICE)
        crypten.save_from_party(samples_bob, "encrypted_data/samples_bob.pt", src=BOB)


    @mpc.run_multiprocess(world_size=2)
    def encryptAndInference():
        # Alice loads some samples, Bob loads other samples
        samples_alice_enc = crypten.load_from_party("encrypted_data/samples_alice.pt", src=ALICE)
        samples_bob_enc = crypten.load_from_party("encrypted_data/samples_bob.pt", src=BOB)

        rank = comm.get().get_rank()
        #crypten.print(f"Rank {rank}: {samples_alice_enc}", in_order=True)

        # Concatenate features
        samples_enc = crypten.cat([samples_alice_enc, samples_bob_enc], dim=0)
        Y_logits = []
        Y_hat = []
        logits, _, y_hat = serverModel(samples_enc, yTemp1, task=task)  # y_hat: (N, T)
        Y_logits += logits.get_plain_text().numpy().tolist()
        y_temp = [np.argmax(item) for item in y_hat.get_plain_text().numpy().tolist()]
        Y_hat.extend(y_temp)
        crypten.print(f"predict:{Y_hat}")

    saveData(input_embeddings)
    encryptAndInference()
