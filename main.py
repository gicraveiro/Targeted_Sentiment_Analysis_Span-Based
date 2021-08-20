#import bert.tokenization as tokenizer
import torch
import transformers # pytorch transformers
import pandas
import numpy
import math
import random
from transformers import AutoConfig, AdamW, get_linear_schedule_with_warmup
#from reused import BertConfig, BertForSpanAspectExtraction
#from reused import run_train_epoch, read_eval_data, read_train_data, prepare_optimizer

def restart_sampling(batch_size):
  # TOKENIZATION
  tokenized_dataset = dataframe['text'].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))#, max_length=100, truncation=True, padding=False''' )
  labels_list = dataframe['labels'].to_list()
  tokenized_dataframe = tokenized_dataset.to_frame()
  tokenized_dataframe.insert(1, "Labels", labels_list, True)

  # RANDOM BATCH REORDERING

  
  dynamic_dataframe = tokenized_dataset.copy(deep=True) #copy of the sentences column of the dataset to delete it parts
  dynamic_labels = dataframe['labels']
  random_batches_list = []
  random_labels_list = []

  while len(dynamic_dataframe) != 0:
    random_index = random.randint(0, len(dynamic_dataframe))
    if (random_index + batch_size >= len(dynamic_dataframe)):
      random_index = random_index - (random_index + batch_size - len(dynamic_dataframe) +1)
      if( random_index < 0):
        random_index = 0
        batch_size = len(dynamic_dataframe)

    batch = dynamic_dataframe[random_index:(random_index+batch_size)]
    batch_labels = dynamic_labels[random_index:(random_index+batch_size)]
    random_batches_list.append(batch)
    random_labels_list.append(batch_labels)

    dynamic_dataframe.drop(dynamic_dataframe.index[random_index:random_index+batch_size], inplace=True)
    dynamic_labels.drop(dynamic_labels.index[random_index:random_index+batch_size], inplace=True)

  # PADDING AND ATTENTION MASK WITH SMART BATCHING

  attention_mask = []
  input_ids = []
  start_positions = []
  end_positions = []

  for batch,labels in zip(random_batches_list,random_labels_list):
    max_len = 0
    for sentence in batch:
      padded_batch = []
      batch_attention_mask = []
      batch_start_positions = []
      batch_end_positions = []

      if (len(sentence) > max_len):
        max_len = len(sentence)
    
    for sentence, sent_label in zip(batch,labels):
      sentence_start_positions = []
      sentence_end_positions = []
      num_zeros = max_len - len(sentence)
      sentence_attention_mask = (len(sentence)*[1] + num_zeros*[0])
      sent_label_list = sent_label.split()
      
      label_i = -1
      #for index in enumerate(sent_label_list):
      for token_i,tok in enumerate(sentence):
        
        if label_i < 0 or label_i >= len(sent_label_list):
          tag = 'O'    
        else:
          tag = sent_label_list[label_i].split("=")
          tag = tag[1]

        if (tokenizer.convert_ids_to_tokens(sentence[token_i])[0:2] == '##'):
          tag = sent_label_list[len(sent_label_list)-1].split("=")
          tag = tag[1]
          truncated_word = 1
          label_i -=1
        else:
          truncated_word = 0   

        if ( tag != 'O' and truncated_word == 0 and (label_i <= 0 or sent_label_list[label_i-1].split("=")[1] == 'O')):
          sentence_start_positions += [1]
        else:
          sentence_start_positions += [0]
        
        if(label_i+1 >= len(sent_label_list)):
          if(tag == 'O' or token_i+1 >= len(sentence)): # list of tokens ended
            sentence_end_positions += [0]
          else: 
            sentence_end_positions += [1]
        elif ( tag != 'O' and ( (sent_label_list[label_i+1].split("=")[1] == 'O' and tokenizer.convert_ids_to_tokens(sentence[token_i+1])[0:2] != '##' ) )): #decision on how to label: words or tokens
          sentence_end_positions += [1]
        else:
          sentence_end_positions += [0]

        label_i += 1

      sentence_start_positions += (num_zeros*[0]) # initial and final token must be added as extra zeros eve beyond the zeros that represent absence of tokens
      sentence_end_positions += (num_zeros*[0])
      sentence = sentence + [0] * num_zeros
      
      padded_batch.append(sentence)
      batch_attention_mask.append(sentence_attention_mask)
      batch_start_positions.append(sentence_start_positions)
      batch_end_positions.append(sentence_end_positions)

    
    input_ids.append(torch.tensor(padded_batch))
    attention_mask.append(torch.tensor(batch_attention_mask))
    start_positions.append(torch.tensor(batch_start_positions))
    end_positions.append(torch.tensor(batch_end_positions))

  return(input_ids,attention_mask, start_positions, end_positions)

#config = AutoConfig.from_pretrained(pretrained_model_name_or_path='distilbert-base-uncased')#,num_labels=2)

qa_model_class, tokenizer_class, pretrained_weights = (transformers.DistilBertForQuestionAnswering, transformers.DistilBertTokenizer, 'distilbert-base-uncased') # for QA 'distilbert-base-uncased-distilled-squad'

# Load pretrained model/tokenizer
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = qa_model_class.from_pretrained(pretrained_weights) 

# Creates a table separating sentences from associated token tags
dataframe = pandas.read_csv("data/laptop14_train.txt", delimiter='####', header=None, names=['text','labels'],engine='python')
tokenized_dataset = dataframe['text'].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))

# Sorts table and transforms each word to the code of the token

new_index_list = dataframe['text'].str.len().sort_values().index
dataframe = dataframe.reindex(new_index_list) # sorted dataframe by length of the sentence
dataframe = dataframe.reset_index(drop=True)

#input_ids, attention_mask, start_positions, end_positions = restart_sampling()

#tokens = input_ids[torch.argmax(start_positions): torch.argmax(end_positions) + 1]
#answerTokens = tokenizer.convert_ids_to_tokens(tokens, skip_special_tokens=True)
#answer_test = tokenizer.convert_tokens_to_string(answerTokens)
#print(answer_test)

# DATA PREPROCESSED IN BERT FORMAT 
#print(input_ids)
#print(attention_mask)
#print(start_positions)
#print(end_positions)

# REUSED FROM PAPER 

#bert_config = BertConfig.from_json_file("bert/bert_config.json") # include bert directory ONLY in local repository

# tokenization as included in the paper code...
#tokenizer = tokenizer.FullTokenizer(vocab_file="bert/vocab_file.txt", do_lower_case=True)
#model = BertForSpanAspectExtraction(bert_config)
# verificação de caminho válido era aqui
#model = bert_load_state_dict(model, torch.load("bert/pytorch_model.bin", map_location='cpu')) # TO DO: EXTRACT FUNCTION FROM PAPER CODE
#print("Loading model from pretrained checkpoint: {}".format("bert/pytorch_model.bin"))

#device = torch.device('cuda') #device = "cpu"
device = "cpu"
model.to(device)

optimizer = AdamW(model.parameters(),
                  lr = 2e-5, # This is the value the paper used
                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                )
epochs_qnt = 1 # TO DO: CHANGE TO 3
batch_size = 8

training_steps = epochs_qnt * math.ceil(len(tokenized_dataset)/batch_size)

#num warmup steps is default value in glue.py
#scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0,num_training_steps = training_steps)
for epoch in range(0,epochs_qnt):

  input_ids, attention_mask, start_positions, end_positions = restart_sampling(batch_size=batch_size)
  print(math.ceil(len(tokenized_dataset)/batch_size),'vs',len(input_ids))
'''
print("***** Preparing data *****")
train_dataloader, num_train_steps = None, None
eval_examples, eval_features, eval_dataloader = None, None, None
train_batch_size = 4 

# their tokenizer was the FullTokenizer but let's head it with this one for now and see what happens
print("***** Preparing training *****")
train_dataloader, num_train_steps = read_train_data(tokenizer)

print("***** Preparing evaluation *****")
eval_examples, eval_features, eval_dataloader = read_eval_data(tokenizer)
print("***** Preparing optimizer *****")
optimizer, param_optimizer = prepare_optimizer(model, num_train_steps)

print("***** Running training *****")
best_f1 = 0
save_checkpoints_steps = int(num_train_steps / (5 * 3)) # 3 = num_train_epochs
start_save_steps = int(num_train_steps * 0.5) # 0.5 = save proportion
model.train()
global_step = 0
'''

#for epoch in range(3):
#  print("***** Epoch: {} *****".format(epoch+1))
#  global_step, model, best_f1 = run_train_epoch(global_step, model, param_optimizer, train_dataloader, eval_examples, eval_features, eval_dataloader, optimizer, 0, device, 'out/extract/01/performance.txt', 'out/extract/01/checkpoint.pth.tar', save_checkpoints_steps, start_save_steps, best_f1) #n_gpu = 0
'''
print("***** Running prediction *****")
        if eval_dataloader is None:
            eval_examples, eval_features, eval_dataloader = read_eval_data(args, tokenizer, logger)

        # restore from best checkpoint
        if save_path and os.path.isfile(save_path) and args.do_train:
            checkpoint = torch.load(save_path)
            model.load_state_dict(checkpoint['model'])
            step = checkpoint['step']
            logger.info("Loading model from finetuned checkpoint: '{}' (step {})"
                        .format(save_path, step))

        model.eval()
        metrics = evaluate(args, model, device, eval_examples, eval_features, eval_dataloader, logger, write_pred=True)
        f = open(log_path, "a")
        print("threshold: {}, step: {}, P: {:.4f}, R: {:.4f}, F1: {:.4f} (common: {}, retrieved: {}, relevant: {})"
              .format(args.logit_threshold, global_step, metrics['p'], metrics['r'],
                      metrics['f1'], metrics['common'], metrics['retrieved'], metrics['relevant']), file=f)
        print(" ", file=f)
        f.close()
        '''


#print(input_ids)
#print(attention_mask)

'''
Acknowledgments

Implementation of the architecture proposed by:

@inproceedings{hu2019open,
  title={Open-Domain Targeted Sentiment Analysis via Span-Based Extraction and Classification},
  author={Hu, Minghao and Peng, Yuxing and Huang, Zhen and Li, Dongsheng and Lv, Yiwei},
  booktitle={Proceedings of ACL},
  year={2019}
}

Smart batching reference:

http://mccormickml.com/2020/07/29/smart-batching-tutorial/#uniform-length-batching

First steps reference:

#code extracted from tutorial at http://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/
''' 