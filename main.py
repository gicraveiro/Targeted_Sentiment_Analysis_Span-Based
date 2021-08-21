#import bert.tokenization as tokenizer
import torch
import transformers # pytorch transformers
import pandas
import numpy
import math
import random
import os
from transformers import AutoConfig, AdamW, get_linear_schedule_with_warmup#BertConfig
from transformers.utils.dummy_pt_objects import LongformerForQuestionAnswering
from reused import BertConfig, BertForSpanAspectExtraction, BERTAdam, evaluate
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
  segment_ids = []

  for batch,labels in zip(random_batches_list,random_labels_list):
    max_len = 0
    for sentence in batch:
      padded_batch = []
      batch_attention_mask = []
      batch_start_positions = []
      batch_end_positions = []
      batch_segment_ids = []

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
      batch_segment_ids = max_len * [0]
      padded_batch.append(sentence)
      batch_attention_mask.append(sentence_attention_mask)
      batch_start_positions.append(sentence_start_positions)
      batch_end_positions.append(sentence_end_positions)

    
    input_ids.append(torch.tensor(padded_batch, dtype=torch.long))
    attention_mask.append(torch.tensor(batch_attention_mask, dtype=torch.long))
    start_positions.append(torch.tensor(batch_start_positions, dtype=torch.long))
    end_positions.append(torch.tensor(batch_end_positions, dtype=torch.long))
    segment_ids.append(torch.tensor(batch_segment_ids, dtype=torch.long))
    #print("start", torch.tensor(batch_start_positions).shape)
    #print("end", torch.tensor(batch_end_positions).shape)

  
  

  return(input_ids,segment_ids,attention_mask, start_positions, end_positions)

def custom_optimizer(model, num_train_steps):
  param_optimizer = list(model.named_parameters())
  #print(param_optimizer)
  no_decay = ['bias', 'LayerNorm']
  optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
  optimizer = BERTAdam(optimizer_grouped_parameters,
                        lr=2e-5,
                        warmup=0.1,
                        t_total=num_train_steps)
  return optimizer, param_optimizer

#os.makedirs('output')
save_path = os.path.join('output', 'checkpoint.pth.tar')
log_path = os.path.join('output', 'performance.txt')
network_path = os.path.join('output', 'network.txt')
parameter_path = os.path.join('output', 'parameter.txt')

#config = AutoConfig.from_pretrained(pretrained_model_name_or_path='distilbert-base-uncased')#,num_labels=2)
#bert_config = BertConfig.from_json_file(args.bert_config_file)
#config = BertConfig(vocab_size=30522)
config = BertConfig.from_json_file("bert/bert_config.json") # include bert directory ONLY in local repository

qa_model_class, tokenizer_class, pretrained_weights = (transformers.DistilBertForQuestionAnswering, transformers.DistilBertTokenizer, 'distilbert-base-uncased-distilled-squad') # for QA 'distilbert-base-uncased-distilled-squad' 'distilbert-base-uncased' 'pt-tinybert-msmarco'
#model_class, tokenizer_class, pretrained_weights = (transformers.BertModel, transformers.BertTokenizer, 'bert-base-uncased')


# Load pretrained model/tokenizer
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
#model = qa_model_class.from_pretrained(pretrained_weights) 
model = BertForSpanAspectExtraction(config)


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


#checkpoint = torch.load(save_path)
#model.load_state_dict(checkpoint['model'])
#step = checkpoint['step']
#print("Loading model from finetuned checkpoint: '{}' (step {})"
#            .format(save_path, step))

f = open(network_path, "w")
for n, param in model.named_parameters():
    print("name: {}, size: {}, dtype: {}, requires_grad: {}"
          .format(n, param.size(), param.dtype, param.requires_grad), file=f)
total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print("Total trainable parameters: {}".format(total_trainable_params), file=f)
print("Total parameters: {}".format(total_params), file=f)
f.close()
eval_examples, eval_features, eval_dataloader = None, None, None
#train dataloader, num trainsteps none


# tokenization as included in the paper code...
#tokenizer = tokenizer.FullTokenizer(vocab_file="bert/vocab_file.txt", do_lower_case=True)
#model = BertForSpanAspectExtraction(bert_config)
# verificação de caminho válido era aqui
#model = bert_load_state_dict(model, torch.load("bert/pytorch_model.bin", map_location='cpu')) # TO DO: EXTRACT FUNCTION FROM PAPER CODE
#print("Loading model from pretrained checkpoint: {}".format("bert/pytorch_model.bin"))

epochs_qnt = 1 # TO DO: CHANGE TO 3
batch_size = 8
training_steps = epochs_qnt * math.ceil(len(tokenized_dataset)/batch_size)
 # This is the learning rate the paper used # args.adam_epsilon  - default is 1e-8.
#optimizer = AdamW(model.parameters(),lr = 2e-5,eps = 1e-8  )
optimizer, param_optimizer = custom_optimizer(model, training_steps)

#device = torch.device('cuda') #device = "cpu"
device = "cpu"
model.to(device)

#checkpoint = torch.load(save_path)
#optimizer.load_state_dict(checkpoint['optimizer'])
#step = checkpoint['step']
#print("Loading optimizer from finetuned checkpoint: '{}' (step {})".format(save_path, step))
global_step = 0 #step

best_f1 = 0
start_save_steps = int(training_steps * 0.5)
save_checkpoints_steps = int(training_steps / (5 * epochs_qnt))

model.train() # only sets the training mode
#num warmup steps is default value in glue.py
#scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0,num_training_steps = training_steps)
for epoch in range(0,epochs_qnt):

  input_ids, segment_ids, attention_mask, start_positions, end_positions = restart_sampling(batch_size=batch_size)
  #print(type(input_ids), type(segment_ids), type(attention_mask), type(start_positions), type(end_positions))

  train_loss = 0.0
  count = 0

  for batch_index,(ids, seg, mask, start, end) in enumerate(zip(input_ids, segment_ids, attention_mask, start_positions, end_positions)):
    #print(batch)

    #input_ids = ids.to(device) # TO DO: CONFIRM THAT IT IS USEFUL TO MOVE THEM TO CPU, OR DELETE IF ITS NOT DOING ANYTHING
    #input_mask = mask.to(device)
    #input_start = start_positions[index].to(device)
    #input_end = end_positions[index].to(device)

    #model.zero_grad() #clear previous gradients

    #print(input_ids.shape,type(input_ids))
    # TRAINING STEP
    #loss is returned because it is supervised learning based on the labels
    # logits are the predicted outputs by the model before activation

    # TO DO: find out what are segment_ids and create them!
    #print(type(train_loss))
    #model = BertForSpanAspectExtraction(bert_config)
    #loss, start_logits, end_logits = model(input_ids=input_ids,  attention_mask=input_mask, start_positions=input_start, end_positions=input_end)
    #print(type(ids), type(seg), type(mask), type(start), type(end))
    loss = model(ids, seg, mask, start, end)
    #start_logits, end_logits = model(ids, seg, mask)
    #score_start, score_end = model(input_ids, input_mask)
    #tokens = inputIds[torch.argmax(scores_start): torch.argmax(score_end) + 1]
    #answerTokens = distilBertTokenizer.convert_ids_to_tokens(tokens, skip_special_tokens=True)
    #print(tokenizer.convert_tokens_to_string(answerTokens))
    # extra layer
    #all_encoder_layers, _ = self.bert(input_ids, segment_ids, attention_mask)
    #sequence_output = all_encoder_layers[-1]
    #self.qa_outputs = nn.Linear(config.hidden_size, 2)
    #logits = self.qa_outputs(sequence_output)
    #start_logits, end_logits = logits.split(1, dim=-1)
    #start_logits = start_logits.squeeze(-1)
    #end_logits = end_logits.squeeze(-1)

    #print(loss)
    loss.backward() # backward propagate
    train_loss += loss.item()
    if numpy.isnan(train_loss) == False:
      print(type(train_loss), train_loss)

    # Clip the norm of the gradients to 1.0.
    # This is to help prevent the "exploding gradients" problem.
    #torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    optimizer.step() #update parameters 
    model.zero_grad()
    global_step += 1
    count += 1
'''
    if global_step % save_checkpoints_steps == 0 and count != 0:
      print("step: {}, loss: {:.4f}".format(global_step, train_loss / count))

    if global_step % save_checkpoints_steps == 0 and global_step > start_save_steps and count != 0:  # eval & save model
      print("***** Running evaluation *****")
      model.eval()
      metrics = evaluate(model, device, eval_examples, eval_features, eval_dataloader)
      f = open("output/performance.txt", "a")
      print("step: {}, loss: {:.4f}, P: {:.4f}, R: {:.4f}, F1: {:.4f} (common: {}, retrieved: {}, relevant: {})"
            .format(global_step, train_loss / count, metrics['p'], metrics['r'],
                    metrics['f1'], metrics['common'], metrics['retrieved'], metrics['relevant']), file=f)
      print(" ", file=f)
      f.close()
      train_loss, count = 0.0, 0
      model.train()
      if metrics['f1'] > best_f1:
          best_f1 = metrics['f1']
          torch.save({
              'model': model.state_dict(),
              'optimizer': optimizer.state_dict(),
              'step': global_step
          }, "output/test")


    #scheduler.step() #update learning rate
    #print(input_ids, input_mask, input_start, input_end)
    #print(batch.to(device))
    '''

  #print("SCORE START", score_start)
  #print("SCORE END", score_end)
print("Total loss", train_loss)
print("Average loss", train_loss/math.ceil(len(tokenized_dataset)/batch_size))
print("Congrats!Training concluded successfully!")
'''
#EVALUATION

  # PREPARE DATASET ON TEST
  # need to send the right parameters!
  input_ids, attention_mask, start_positions, end_positions = restart_sampling(batch_size=batch_size)

  model.eval() #set to evaluation mode

  predictions, start_labels, end_labels = [], []

  for index, batch in enumerate(input_ids):

    input_ids = batch.to(device) # TO DO: CONFIRM THAT IT IS USEFUL TO MOVE THEM TO CPU, OR DELETE IF ITS NOT DOING ANYTHING
    input_mask = attention_mask[index].to(device)
    input_start = start_positions[index].to(device)
    input_end = end_positions[index].to(device)

    with torch.no_grad():
      outputs = model(input_ids, token_type_ids=None, attention_mask=input_mask) # check token_type_ids
      
    logits = outputs[0]

    # Move logits and labels to CPU
    logits = logits.detach().cpu().numpy()
    start_ids = input_start.to('cpu').numpy()
    end_ids = input_end.to('cpu').numpy()

    predictions.append(logits)
    start_labels.append(start_ids)
    end_labels.append(end_ids)

'''
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