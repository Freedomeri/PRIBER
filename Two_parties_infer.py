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
import torch.nn.functional as nn
import collections


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

    embeddings = inputs_embeds + position_embeddings + token_type_embeddings
    embeddings = nn.layer_norm(embeddings,tuple(hidden_size),weight = BertEmbeds['bert.embeddings.LayerNorm.weight'], bias = BertEmbeds['bert.embeddings.LayerNorm.bias'],eps=1e-12)
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



if __name__ == "__main__":

    ALICE = 0
    BOB = 1

    '''static configs'''
    use_gpu = False
    lm = 'bert-small'
    task = 'wdc_all_small'
    max_len = 128
    model_path = "/home/smz/models/bert-small-finetuned-wdcs2"
    input_path = 'input/input_wdc_all_10.txt'

    '''Split saved Parameters into embeddings and main model params'''
    BertEmbeds, saved_state = splitModel(model_path)

    '''load server model'''
    taskConfig,serverModel = load_model(task,saved_state,lm,use_gpu,path = model_path)
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
            input_embeddings.append(embedding(x, BertEmbeds, hidden_size=512))
            Y.append(y)


    '''divide into 2 parts(for test)'''
    # temp1 = input_embeddings[0][:75]
    # temp2 = input_embeddings[0][75:]
    # yTemp1 = Y[0][:75]
    # yTemp2 = Y[0][75:]
    # input_embeddings[0] = temp1
    # input_embeddings.append(temp2)
    # yTemp1 = Y[0]
    # yTemp2 = Y[1]
    samples_bob = input_embeddings[0]

    '''encrypt and process for 2 parties'''
    from crypten import mpc
    import crypten.communicator as comm
    torch.set_num_threads(1)

    @mpc.run_multiprocess(world_size=2)
    def saveData(input_embeddings):
        #samples_alice = input_embeddings[0]

        """encrypt data"""
        #crypten.save_from_party(samples_alice, "encrypted_data/samples_alice.pt", src=ALICE)
        crypten.save_from_party(samples_bob, "encrypted_data/samples_bob.pt", src=BOB)


        """encrypt model"""
        model = crypten.load_from_party(saved_model_path, model_class=MultiTaskNet, src=ALICE)
        encrypted_model = crypten.nn.from_pytorch(model, torch.empty(1, 128, 512),
                                                  transformers=False)
        encrypted_model.encrypt(src=ALICE)
        '''Check that model is encrypted:'''
        crypten.print("Model successfully encrypted:", encrypted_model.encrypted)


    #@mpc.run_multiprocess(world_size=2)
    #def encryptAndInference():
        ''' Alice loads model, Bob loads samples'''
        #samples_alice_enc = crypten.load_from_party("encrypted_data/samples_alice.pt", src=ALICE)
        samples_enc = crypten.load_from_party("encrypted_data/samples_bob.pt", src=BOB)
        encrypted_model.eval()

        rank = comm.get().get_rank()
        #crypten.print(f"Rank {rank}: {samples_alice_enc}", in_order=True)

        # Concatenate features
        #samples_enc = crypten.cat([samples_alice_enc, samples_bob_enc], dim=0)
        #Y_logits = []
        Y_hat = []
        #logits, _, y_hat = serverModel(samples_enc, yTemp1, task=task)  # y_hat: (N, T)
        #y_hat = encrypted_model(samples_enc, Y[0], task=task)  # y_hat: (N, T)
        y_hat = encrypted_model(samples_enc)  # y_hat: (N, T)
        #Y_logits += logits.get_plain_text().numpy().tolist()
        #y_temp = [np.argmax(item) for item in y_hat.get_plain_text().numpy().tolist()]
        #Y_hat.extend(y_temp)
        y_hat = y_hat.argmax(-1).get_plain_text().numpy().tolist()
        Y_hat.extend(y_hat)
        crypten.print(f"predict:{Y_hat}")

    def testOneParty(input_embeddings):
        samples_test = crypten.cryptensor(input_embeddings[0])
        plaintext_model = torch.load(saved_model_path)
        encrypted_model = crypten.nn.from_pytorch(plaintext_model, torch.empty((1, 128, 512)),
                                                  transformers=False)
        encrypted_model.eval()
        encrypted_model.encrypt()
        crypten.print("Model successfully encrypted:", encrypted_model.encrypted)
        y_hat = encrypted_model(samples_test)  # y_hat: (N, T)
        y_hat = y_hat.argmax(-1)
        Y_hat = [np.argmax(item) for item in y_hat.get_plain_text().numpy().tolist()]
        crypten.print(f"predict:{Y_hat}")


    testOneParty(input_embeddings)

    #saveData(input_embeddings)
    #encryptAndInference()
