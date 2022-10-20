import numpy
from apex import amp
import torch
import numpy as np
import crypten
import transformers.modeling_bert
from ditto.exceptions import ModelNotFoundError
crypten.init()
import json
import jsonlines
import sys
import time
import timeit
import argparse
from tqdm import tqdm
sys.path.insert(0, "Snippext_public")
from torch.utils import data
from snippext.model import MultiTaskNet
from snippext.model import EncryptedNet
from ditto.exceptions import ModelNotFoundError
from ditto.dataset import DittoDataset
from ditto.summarize import Summarizer
from ditto.knowledge import *
import os
import MyBertModel
import math
import torch.nn.functional as nn
import collections
import sklearn.metrics as metrics
import logging
logging.getLogger().setLevel(logging.INFO)
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

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

def load_model(task, saved_state, lm, use_gpu, fp16=False,path = None, hidden_size=768):
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
    model = EncryptedNet([config], device, False, lm=lm,bert_path = path, task = task,hidden_size=hidden_size)

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
                writer.write(line.split('\t')[:3])
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

def write_results(rows, predictions, output_path):
    with jsonlines.open(output_path, mode='w') as writer:
        for row, pred in zip(rows, predictions):
            output = {'pair': row,
                'match': int(pred)}
            writer.write(output)

def embedding(input_ids, BertEmbeds, position_ids = None, token_type_ids = None, inputs_embeds = None, hidden_size = 512):
    if input_ids is not None:
        input_shape = input_ids.size()
    else:
        input_shape = inputs_embeds.size()[:-1]
    hidden_size = (hidden_size,)

    seq_length = input_shape[1]
    device = input_ids.device if input_ids is not None else inputs_embeds.device
    if position_ids is None:
        position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(input_shape)
    if token_type_ids is None:
        token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

    if inputs_embeds is None:
        inputs_embeds = nn.embedding(input_ids,BertEmbeds['bert.embeddings.word_embeddings.weight'])
    position_embeddings = nn.embedding(position_ids,BertEmbeds['bert.embeddings.position_embeddings.weight'])
    token_type_embeddings = nn.embedding(token_type_ids,BertEmbeds['bert.embeddings.token_type_embeddings.weight'])
    #import matplotlib.pyplot as plt
    embeddings = inputs_embeds + position_embeddings + token_type_embeddings
    # test1 = embeddings.flatten()
    # plt.plot(test1)
    # plt.show()
    embeddings = nn.layer_norm(embeddings,tuple(hidden_size),weight = BertEmbeds['bert.embeddings.LayerNorm.weight'], bias = BertEmbeds['bert.embeddings.LayerNorm.bias'],eps=1e-12)
    # test2 = embeddings.flatten()
    # plt.plot(test2)
    # plt.show()
    embeddings = nn.dropout(embeddings,p = 0.1,training=False)

    return embeddings

def splitModel(path):
    path = os.path.join(path, 'pytorch_model.bin')
    if not os.path.exists(path):
        raise ModelNotFoundError(path)
    saved_state = torch.load(path, map_location=lambda storage, loc: storage)
    BertEmbeds = collections.OrderedDict()
    BertEmbeds['bert.embeddings.word_embeddings.weight'] = saved_state.pop('bert.embeddings.word_embeddings.weight')
    BertEmbeds['bert.embeddings.position_embeddings.weight'] = saved_state.pop('bert.embeddings.position_embeddings.weight')
    BertEmbeds['bert.embeddings.token_type_embeddings.weight'] = saved_state.pop('bert.embeddings.token_type_embeddings.weight')
    BertEmbeds['bert.embeddings.LayerNorm.weight'] = saved_state.pop('bert.embeddings.LayerNorm.weight')
    BertEmbeds['bert.embeddings.LayerNorm.bias'] = saved_state.pop('bert.embeddings.LayerNorm.bias')
    return BertEmbeds, saved_state

def registerSafeClass():
    crypten.common.serial.register_safe_class(MyBertModel.MyBERTModel)
    crypten.common.serial.register_safe_class(transformers.modeling_bert.BertEncoder)
    crypten.common.serial.register_safe_class(transformers.modeling_bert.BertSelfAttention)
    crypten.common.serial.register_safe_class(transformers.modeling_bert.BertIntermediate)
    crypten.common.serial.register_safe_class(transformers.modeling_bert.BertOutput)
    crypten.common.serial.register_safe_class(transformers.modeling_bert.BertLayer)
    crypten.common.serial.register_safe_class(transformers.modeling_bert.BertAttention)
    crypten.common.serial.register_safe_class(transformers.modeling_bert.BertSelfOutput)
    crypten.common.serial.register_safe_class(transformers.configuration_bert.BertConfig)
    crypten.common.serial.register_safe_class(transformers.modeling_bert.BertPooler)
    crypten.common.serial.register_safe_class(torch.nn.modules.container.ModuleDict)
    crypten.common.serial.register_safe_class(transformers.activations.Activation)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="Structured_Beer")
    parser.add_argument("--run_id", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--logdir", type=str, default="checkpoints/")
    parser.add_argument("--lm", type=str, default='bert')
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--input_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--fp16", dest="fp16", action="store_true")
    parser.add_argument("--alpha_aug", type=float, default=0.8)
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--use_gpu", type=bool, default=False)

    hp = parser.parse_args()

    ALICE = 0
    BOB = 1

    '''static configs'''
    use_gpu = hp.use_gpu
    lm = hp.lm
    task = hp.task
    batch_size = hp.batch_size
    max_len = hp.max_len
    hidden_size = hp.hidden_size
    model_path = hp.model_path
    input_path = hp.input_path
    output_path = hp.output_path


    '''Split saved Parameters into embeddings and main model params'''
    BertEmbeds, saved_state = splitModel(model_path)

    '''load server model'''
    taskConfig,serverModel = load_model(task,saved_state,lm,use_gpu,path = model_path,hidden_size=hidden_size)
    saved_model_path = os.path.join(os.getcwd(),'models','serverModel','pytorch_model.bin')
    #serverModel.save_pretrained(saved_model_path)
    torch.save(serverModel,saved_model_path)
    registerSafeClass()


    '''load client dataset'''
    dataPairs,dataRows = load_dataset(input_path,taskConfig,max_len=max_len)
    inputs = []
    for (sentA, sentB) in dataPairs:
        inputs.append(sentA + '\t' + sentB)

    dataset = DittoDataset(inputs, taskConfig['vocab'], taskConfig['name'], lm=lm, max_len=max_len)
    iterator = data.DataLoader(dataset=dataset,
                               batch_size=len(dataset),
                               shuffle=False,
                               num_workers=0,
                               collate_fn=DittoDataset.pad)
    input_embeddings = []

    '''load labels'''
    Y = []
    for label in dataRows:
        Y.append(int(label[2]))

    '''embed locally'''
    with torch.no_grad():
        # print('Classification')
        for i, batch in enumerate(iterator):
            word, x, is_heads, tags, mask, y, seqlens, taskname = batch
            #taskname = taskname[0]
            input_embeddings.append(embedding(x, BertEmbeds, hidden_size=hidden_size))
            #Y.append(y)


    '''batch'''
    batch_num = math.ceil(input_embeddings[0].__len__() / batch_size)
    samples_bob = input_embeddings[0]

    '''encrypt and process for 2 parties'''
    from crypten import mpc
    import crypten.communicator as comm
    torch.set_num_threads(1)


    @mpc.run_multiprocess(world_size=2)
    def saveData(task_name):

        """encrypt data"""
        crypten.save_from_party(samples_bob, "encrypted_data/samples_bob.pt", src=BOB)

        """encrypt model"""
        model = crypten.load_from_party(saved_model_path, model_class=EncryptedNet, src=ALICE)
        encrypted_model = crypten.nn.from_pytorch(model, torch.empty(1, max_len, hidden_size),
                                                  transformers=False)
        encrypted_model.encrypt(src=ALICE)
        '''Check that model is encrypted:'''
        crypten.print("Model successfully encrypted:", encrypted_model.encrypted)

        ''' Alice loads model, Bob loads samples'''
        #samples_alice_enc = crypten.load_from_party("encrypted_data/samples_alice.pt", src=ALICE)
        samples_enc = crypten.load_from_party("encrypted_data/samples_bob.pt", src=BOB)
        encrypted_model.eval()

        rank = comm.get().get_rank()
        eva_time = 0
        eva_bytes = 0
        Y_hat = []
        for _batch in range(batch_num):
            start, end = _batch * batch_size, (_batch + 1) * batch_size
            if end > samples_enc.__len__():
                end = samples_enc.__len__()
            crypten.reset_communication_stats()
            t_start = timeit.default_timer()
            y_hat = encrypted_model(samples_enc[start:end])  # y_hat: (N, T)
            t_end = timeit.default_timer()
            eva_time += t_end-t_start
            eva_bytes += comm.get().get_communication_stats().get('bytes')
            crypten.print(f"Evaluation time:{t_end - t_start}")
            y_hat = y_hat.argmax(-1).get_plain_text().cpu().numpy().tolist()
            y_hat = [np.argmax(item) for item in y_hat]
            Y_hat.extend(y_hat)
            crypten.print(f"predict:{Y_hat}")
            crypten.print_communication_stats()


        crypten.print(f"Total Evaluation time:{eva_time}")
        crypten.print(f"Total Communication Bytes:{eva_bytes}")
        accuracy = metrics.accuracy_score(Y, Y_hat)
        precision = metrics.precision_score(Y, Y_hat)
        recall = metrics.recall_score(Y, Y_hat)
        f1 = metrics.f1_score(Y, Y_hat)
        crypten.print("accuracy=%.3f" % accuracy)
        crypten.print("precision=%.3f" % precision)
        crypten.print("recall=%.3f" % recall)
        crypten.print("f1=%.3f" % f1)
        crypten.print("======================================")
        return Y_hat

    def testOneParty(input_embeddings):
        samples_test = crypten.cryptensor(input_embeddings[0])
        samples_test = samples_test.cuda() if torch.cuda.is_available() else samples_test.cpu()
        plaintext_model = torch.load(saved_model_path)
        encrypted_model = crypten.nn.from_pytorch(plaintext_model, torch.empty((1, 128, hidden_size)),
                                                  transformers=False)
        encrypted_model.eval()
        encrypted_model.encrypt()
        crypten.print("Model successfully encrypted:", encrypted_model.encrypted)
        encrypted_model = encrypted_model.cuda() if torch.cuda.is_available() else encrypted_model.cpu()
        y_hat = encrypted_model(samples_test)  # y_hat: (N, T)
        y_hat = y_hat.argmax(-1)
        Y_hat = [np.argmax(item) for item in y_hat.get_plain_text().cpu().numpy().tolist()]
        crypten.print(f"predict:{Y_hat}")


    #testOneParty(input_embeddings)

    Y = saveData(task)
    print(Y)
    #results = Y[::2]
    #results = list(numpy.array(results).flat)
    #write_results(dataPairs, results, output_path)
    #encryptAndInference()
