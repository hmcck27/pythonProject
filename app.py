from flask import Flask,request
import json
import torch
from model.net import KobertBiGRUCRF
from gluonnlp.data import SentencepieceTokenizer
from data_utils.utils import Config
from data_utils.vocab_tokenizer import Tokenizer
from data_utils.pad_sequence import keras_pad_fn
import pickle

model_dir = './experiments/base_model_with_bigru_crf'
model_config_path = './experiments/base_model_with_bigru_crf/config.json'
vocab_pickle_path = './experiments/base_model_with_bigru_crf/vocab.pkl'
ner_to_index_path = './experiments/base_model_with_bigru_crf/ner_to_index.json'

class DecoderFromNamedEntitySequence():
    def __init__(self, tokenizer, index_to_ner):
        self.tokenizer = tokenizer
        self.index_to_ner = index_to_ner

    def __call__(self, list_of_input_ids, list_of_pred_ids):
        input_token = self.tokenizer.decode_token_ids(list_of_input_ids)[0]
        pred_ner_tag = [self.index_to_ner[pred_id] for pred_id in list_of_pred_ids[0]]

        print("len: {}, input_token:{}".format(len(input_token), input_token))
        print("len: {}, pred_ner_tag:{}".format(len(pred_ner_tag), pred_ner_tag))

        # ----------------------------- parsing list_of_ner_word ----------------------------- #
        list_of_ner_word = []
        entity_word, entity_tag, prev_entity_tag = "", "", ""
        for i, pred_ner_tag_str in enumerate(pred_ner_tag):
            if "B-" in pred_ner_tag_str:
                entity_tag = pred_ner_tag_str[-3:]

                if prev_entity_tag != entity_tag and prev_entity_tag != "":
                    list_of_ner_word.append({"word": entity_word.replace("▁", " "), "tag": prev_entity_tag, "prob": None})

                entity_word = input_token[i]
                prev_entity_tag = entity_tag
            elif "I-"+entity_tag in pred_ner_tag_str:
                entity_word += input_token[i]
            else:
                if entity_word != "" and entity_tag != "":
                    list_of_ner_word.append({"word":entity_word.replace("▁", " "), "tag":entity_tag, "prob":None})
                entity_word, entity_tag, prev_entity_tag = "", "", ""


        # ----------------------------- parsing decoding_ner_sentence ----------------------------- #
        decoding_ner_sentence = ""
        is_prev_entity = False
        prev_entity_tag = ""
        is_there_B_before_I = False

        for token_str, pred_ner_tag_str in zip(input_token, pred_ner_tag):
            token_str = token_str.replace('▁', ' ')  # '▁' 토큰을 띄어쓰기로 교체

            if 'B-' in pred_ner_tag_str:
                if is_prev_entity is True:
                    decoding_ner_sentence += ':' + prev_entity_tag+ '>'

                if token_str[0] == ' ':
                    token_str = list(token_str)
                    token_str[0] = ' <'
                    token_str = ''.join(token_str)
                    decoding_ner_sentence += token_str
                else:
                    decoding_ner_sentence += '<' + token_str
                is_prev_entity = True
                prev_entity_tag = pred_ner_tag_str[-3:] # 첫번째 예측을 기준으로 하겠음
                is_there_B_before_I = True

            elif 'I-' in pred_ner_tag_str:
                decoding_ner_sentence += token_str

                if is_there_B_before_I is True: # I가 나오기전에 B가 있어야하도록 체크
                    is_prev_entity = True
            else:
                if is_prev_entity is True:
                    decoding_ner_sentence += ':' + prev_entity_tag+ '>' + token_str
                    is_prev_entity = False
                    is_there_B_before_I = False
                else:
                    decoding_ner_sentence += token_str

        return list_of_ner_word, decoding_ner_sentence


## global variable
global ner_tokenizer
global decoder_from_res
global ner_model

def load_preprocessing_model():
    ## TO-DO
    ## load 개체명 인식기 모델

    model_config = Config(model_config_path)

    # Vocab & Tokenizer

    tok_path = "./ptr_lm_model/tokenizer_78b3253a26.model"
    ptr_tokenizer = SentencepieceTokenizer(tok_path)

    # load vocab & tokenizer
    with open(vocab_pickle_path, 'rb') as f:
        vocab = pickle.load(f)

    global ner_tokenizer
    ner_tokenizer = Tokenizer(vocab=vocab, split_fn=ptr_tokenizer, pad_fn=keras_pad_fn, maxlen=model_config.maxlen)

    # load ner_to_index.json
    with open(ner_to_index_path, 'rb') as f:
        ner_to_index = json.load(f)
        index_to_ner = {v: k for k, v in ner_to_index.items()}

    # Model
    global ner_model
    ner_model = KobertBiGRUCRF(config=model_config, num_classes=len(ner_to_index), vocab=vocab)

    # load
    model_dict = ner_model.state_dict()
    checkpoint = torch.load("./experiments/base_model_with_bigru_crf/model-epoch-18-step-3250-acc-0.997.bin",
                            map_location=torch.device('cpu'))

    convert_keys = {}
    for k, v in checkpoint['model_state_dict'].items():
        new_key_name = k.replace("module.", '')
        if new_key_name not in model_dict:
            print("{} is not int model_dict".format(new_key_name))
            continue
        convert_keys[new_key_name] = v

    ## 수정
    ner_model.load_state_dict(convert_keys, strict=False)
    ner_model.eval()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    ner_model.to(device)
    global decoder_from_res
    decoder_from_res = DecoderFromNamedEntitySequence(tokenizer=ner_tokenizer, index_to_ner=index_to_ner)

def load_koBERT_model():
    ## TO-DO
    ## load koBERT 모델
    print()

def start_preprocessing(text):

    print('start preprocessing!')
    print("text : ",text)
    # 1. 엔터 탭 없에기
    text = text.replace('\n',' ')
    text = text.replace('\t',' ')

    # 2. 개체명 인식기를 돌려서 이름, 지역, 조직을 없엔다.
    list_of_input_ids = ner_tokenizer.list_of_string_to_list_of_cls_sep_token_ids([text])
    x_input = torch.tensor(list_of_input_ids).long()

    ## for bert alone
    # y_pred = model(x_input)
    # list_of_pred_ids = y_pred.max(dim=-1)[1].tolist()

    ## for bert crf
    ## list_of_pred_ids = model(x_input)

    ## for bert bilstm crf & bert bigru crf
    list_of_pred_ids = ner_model(x_input, using_pack_sequence=False)

    list_of_ner_word, decoding_ner_sentence = decoder_from_res(list_of_input_ids=list_of_input_ids,
                                                               list_of_pred_ids=list_of_pred_ids)

    return decoding_ner_sentence


def model_predict(text):
    print()
app = Flask(__name__)

@app.route('/desc',methods=['POST'])
def predict_algorithm():

    # using post method
    # get json data from front side

    params = request.get_json()
    desc = params['description']
    input = params['input']
    output = params['output']

    # 먼저 전처리 하기
    base_preprocessing(desc)

    # 모델로 결과 예측
    json_result = model_predict(text)

    

if __name__ == "__main__" :
    print('두개의 모델을 로드해야 합니다. 조금 기다려주세요')
    load_preprocessing_model()
    while True:
        text = input()
        print(start_preprocessing(text))

    # load_koBERT_model()
    # app.run(host='0.0.0.0')

