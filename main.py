import torch
import transformers # pytorch transformers
import pandas
import math
import random
import time
from transformers import AutoConfig, AdamW, get_linear_schedule_with_warmup
import numpy

def restart_sampling(batch_size, input_file):

  # Creates a table separating sentences from associated token tags
  dataframe = pandas.read_csv(input_file, delimiter='####', header=None, names=['text','labels'],engine='python')

  tokenized_dataset = dataframe['text'].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))

  # Sorts table and transforms each word to the code of the token

  new_index_list = dataframe['text'].str.len().sort_values().index
  dataframe = dataframe.reindex(new_index_list) # sorted dataframe by length of the sentence
  dataframe = dataframe.reset_index(drop=True)
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
  polarities = []

  for batch,labels in zip(random_batches_list,random_labels_list):
    max_len = 0
    for sentence in batch:
      padded_batch = []
      batch_attention_mask = []
      batch_start_positions = []
      batch_end_positions = []
      batch_polarities = []

      if (len(sentence) > max_len):
        max_len = len(sentence)
    
    for sentence, sent_label in zip(batch,labels):
      sentence_start_positions = []
      sentence_end_positions = []
      sentence_polarities = []
      num_zeros = max_len - len(sentence)
      sentence_attention_mask = (len(sentence)*[1] + num_zeros*[0])
      sent_label_list = sent_label.split()
      
      label_i = -1
      finished = 0
      for token_i,tok in enumerate(sentence):
        
        if label_i < 0 or label_i >= len(sent_label_list):
          tag = 'O'    
        else:
          tag = sent_label_list[label_i].split("=")
          tag = tag[len(tag)-1] 

        end_index = token_i
        if( tag != 'O'):
          tag = tag.split("-")
          tag = tag[1]
          if (tag != 'NEU'):
            sentence_start_positions = [token_i]
            sentence_end_positions = [token_i]
            if(tag == 'POS'):
              sentence_polarities = [0]
            elif(tag == 'NEG'):
              sentence_polarities = [1]
            #elif(tag ==  'NEU'):
            #  sentence_polarities = [2]
        while(tag != 'O' and tag != 'NEU' and label_i < len(sent_label_list)):
          tag = sent_label_list[label_i].split("=")
          tag = tag[1]

          if(tag == 'O'):
            sentence_end_positions = [end_index-1]
            finished = 1
          label_i += 1
          end_index += 1
          
        if (finished == 1):
          break
        label_i +=1
        
      if(sentence_start_positions == []):
        sentence_start_positions = [-1]
        sentence_end_positions = [-1]
        sentence_polarities = [2] # 3 EQUALS TO NO TARGETED SPAN IN THIS SEQUENCE, 2 if neutrals have been filtered
      sentence = sentence + [0] * num_zeros
      padded_batch.append(sentence)
      batch_attention_mask.append(sentence_attention_mask)
      batch_start_positions += sentence_start_positions
      batch_end_positions += sentence_end_positions
      batch_polarities += sentence_polarities
    
    input_ids.append(torch.tensor(padded_batch, dtype=torch.long))
    attention_mask.append(torch.tensor(batch_attention_mask, dtype=torch.long))
    start_positions.append(torch.tensor(batch_start_positions, dtype=torch.long))
    end_positions.append(torch.tensor(batch_end_positions, dtype=torch.long))
    polarities.append(torch.tensor(batch_polarities, dtype=torch.long))

  return(input_ids,attention_mask, start_positions, end_positions, polarities)

initial_time = time.time()

config = AutoConfig.from_pretrained(pretrained_model_name_or_path='distilbert-base-uncased', n_layers=4, hidden_dim=1200, dim=312, max_position_embeddings=312)
#config = BertConfig(vocab_size=30522)
#config = BertConfig.from_json_file("bert/bert_config.json") # include bert directory ONLY in local repository

qa_model_class, tokenizer_class, pretrained_weights = (transformers.DistilBertForQuestionAnswering, transformers.DistilBertTokenizer, 'distilbert-base-uncased') 

# Load pretrained model/tokenizer
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = qa_model_class.from_pretrained(pretrained_weights) 

with open('data/laptop14_train.txt') as file:
  train_dataset_len = sum(1 for line in file)
# TRAINING CONFIGURATIONS

epochs_qnt = 1#3
batch_size = 8
training_steps = epochs_qnt * math.ceil(train_dataset_len/batch_size)

# Setting optimizer
 # This is the learning rate the paper used # args.adam_epsilon  - default is 1e-8.
optimizer = AdamW(model.parameters(),lr = 2e-5,eps = 1e-8  ) #change learning rate

device = "cpu"
model.to(device)

# TRAINING STEP

model.train() # only sets the training mode
#num warmup steps is default value in glue.py
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0,num_training_steps = training_steps)
for epoch in range(0,epochs_qnt):
  
  train_loss = 0.0
  input_ids,  attention_mask, start_positions, end_positions, polarities = restart_sampling(batch_size, "data/laptop14_train.txt")

  for batch_index,(input_ids, input_mask, input_start, input_end) in enumerate(zip(input_ids, attention_mask, start_positions, end_positions)):

    model.zero_grad() #clear previous gradients

    #loss is returned because it is supervised learning based on the labels
    # logits are the predicted outputs by the model before activation
    outputs = model(input_ids=input_ids,  attention_mask=input_mask, start_positions=input_start, end_positions=input_end)
    loss = outputs.loss
    
    loss.backward() # backward propagate
    train_loss += loss.item()

    # Clip the norm of the gradients to 1.0.
    # This is to help prevent the "exploding gradients" problem.
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    optimizer.step() #update parameters 
    scheduler.step()
    #model.zero_grad()

training_time = time.time() - initial_time
print("Congrats! Training concluded successfully!\n")
print("Average loss", train_loss/math.ceil(train_dataset_len/batch_size))
print("Training time in minutes:", training_time/60)

#EVALUATION

# PREPARE DATASET ON TEST
input_ids, attention_mask, start_positions, end_positions, polarities = restart_sampling(batch_size, "data/laptop14_test.txt")

model.eval() # set to evaluation mode

predicted_starts, predicted_ends, real_starts, real_ends = [], [], [], []
count = 0
real_count, pred_count = 0,0

for batch_index,(input_ids, input_mask, input_start, input_end) in enumerate(zip(input_ids, attention_mask, start_positions, end_positions)):

  with torch.no_grad():
    outputs = model(input_ids, attention_mask=input_mask) 

  start_logits = outputs.start_logits
  end_logits = outputs.end_logits

  for logit in start_logits:
    start_logits = logit.numpy()
    end_logits = logit.numpy()
    pred_start = numpy.max(start_logits)
    pred_end = numpy.max(end_logits)
    pred_start = numpy.nonzero(start_logits == pred_start)
    pred_end = numpy.nonzero(end_logits == pred_end)

    predicted_starts.append(pred_start[0][0])
    predicted_ends.append(pred_end[0][0])

  real_starts.append(input_start)
  real_ends.append(input_end)

eval_time = time.time() - training_time - initial_time
print('Congrats! Evaluation concluded successfully!\n')
print('Evaluation time in seconds:', eval_time)

total_real_starts = numpy.concatenate(real_starts, axis=0)
total_real_ends = numpy.concatenate(real_ends, axis=0)
total_real_starts[total_real_starts == -1] = 0
total_real_ends[total_real_ends == -1] = 0
total_real_starts = list(total_real_starts)
total_real_ends = list(total_real_ends)

true_positives = 0
for index, (pred_start, real_start, pred_end, real_end) in enumerate(zip(predicted_starts,total_real_starts, predicted_ends, total_real_ends)):
  if(predicted_starts[index] == total_real_starts[index] and predicted_ends[index] == total_real_ends[index]):
    true_positives += 1

total_time = time.time() - initial_time
print("Total true positives - both span start and end positions:", true_positives)
print("Accuracy", true_positives/len(predicted_starts))
print("Total target extraction time in minutes:", total_time/60)
print("Parameters: ")
print("Number of epochs:", epochs_qnt)
print("Batch size:", batch_size)
print("DistilBERT used instead of BERT")
print("Number of hidden layers: 4")
print("Hidden size: 312")
print("Intermediate size: 1200")
# TO DO: report in a more organized way the EFFICIENCY based on size and time
#
#
#
#
#
#
#
#
# POLARITY CLASSIFICATION
#
#
#
#
#
#
#
# labels = Positive, Negative, -removed Neutral for now- , No target in the span
'''
config = AutoConfig.from_pretrained(pretrained_model_name_or_path='distilbert-base-uncased', n_layers=4, hidden_dim=1200, dim=312, max_position_embeddings=312, num_labels=3) #num_labels=4
class_model = (transformers.DistilBertForSequenceClassification) 

# Load pretrained model/tokenizer
model = class_model.from_pretrained(pretrained_weights) 

# Setting optimizer
 # This is the learning rate the paper used # args.adam_epsilon  - default is 1e-8.
optimizer = AdamW(model.parameters(),lr = 2e-5,eps = 1e-8  ) #change learning rate

device = "cpu"
model.to(device)

# TRAINING STEP

model.train() # only sets the training mode
#num warmup steps is default value in glue.py
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0,num_training_steps = training_steps)
for epoch in range(0,epochs_qnt):
  
  train_loss = 0.0
  input_ids,  attention_mask, start_positions, end_positions, polarities = restart_sampling(batch_size, "data/laptop14_train.txt")

  for batch_index,(input_ids, input_mask, input_start, input_end) in enumerate(zip(input_ids, attention_mask, start_positions, end_positions)):

    model.zero_grad() #clear previous gradients

    #loss is returned because it is supervised learning based on the labels
    # logits are the predicted outputs by the model before activation
    outputs = model(input_ids=input_ids,  attention_mask=input_mask, start_positions=input_start, end_positions=input_end)
    loss = outputs.loss
    
    loss.backward() # backward propagate
    train_loss += loss.item()

    # Clip the norm of the gradients to 1.0.
    # This is to help prevent the "exploding gradients" problem.
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    optimizer.step() #update parameters 
    scheduler.step()
    #model.zero_grad()

training_time = time.time() - initial_time
print("Congrats! Training concluded successfully!\n")
print("Average loss", train_loss/math.ceil(train_dataset_len/batch_size))
print("Training time in minutes:", training_time/60)

#EVALUATION

# PREPARE DATASET ON TEST
input_ids, attention_mask, start_positions, end_positions, polarities = restart_sampling(batch_size, "data/laptop14_test.txt")

model.eval() # set to evaluation mode

predicted_starts, predicted_ends, real_starts, real_ends = [], [], [], []
count = 0
real_count, pred_count = 0,0

for batch_index,(input_ids, input_mask, input_start, input_end) in enumerate(zip(input_ids, attention_mask, start_positions, end_positions)):

  with torch.no_grad():
    outputs = model(input_ids, attention_mask=input_mask) 

  start_logits = outputs.start_logits
  end_logits = outputs.end_logits

  for logit in start_logits:
    start_logits = logit.numpy()
    end_logits = logit.numpy()
    pred_start = numpy.max(start_logits)
    pred_end = numpy.max(end_logits)
    pred_start = numpy.nonzero(start_logits == pred_start)
    pred_end = numpy.nonzero(end_logits == pred_end)

    predicted_starts.append(pred_start[0][0])
    predicted_ends.append(pred_end[0][0])

  real_starts.append(input_start)
  real_ends.append(input_end)

eval_time = time.time() - training_time - initial_time
print('Congrats! Evaluation concluded successfully!\n')
print('Evaluation time in seconds:', eval_time)

total_real_starts = numpy.concatenate(real_starts, axis=0)
total_real_ends = numpy.concatenate(real_ends, axis=0)
total_real_starts[total_real_starts == -1] = 0
total_real_ends[total_real_ends == -1] = 0
total_real_starts = list(total_real_starts)
total_real_ends = list(total_real_ends)

true_positives = 0
for index, (pred_start, real_start, pred_end, real_end) in enumerate(zip(predicted_starts,total_real_starts, predicted_ends, total_real_ends)):
  if(predicted_starts[index] == total_real_starts[index] and predicted_ends[index] == total_real_ends[index]):
    true_positives += 1

total_time = time.time() - initial_time
print("Total true positives - both span start and end positions:", true_positives)
print("Accuracy", true_positives/len(predicted_starts))
print("Total target extraction time in minutes:", total_time/60)
print("Parameters: ")
print("Number of epochs:", epochs_qnt)
print("Batch size:", batch_size)
print("DistilBERT used instead of BERT")
print("Number of hidden layers: 4")
print("Hidden size: 312")
print("Intermediate size: 1200")
# TO DO: report in a more organized way the EFFICIENCY based on size and time



'''

'''
Acknowledgments

Implementation of the architecture proposed by:

@inproceedings{hu2019open,
  title={Open-Domain Targeted Sentiment Analysis via Span-Based Extraction and Classification},
  author={Hu, Minghao and Peng, Yuxing and Huang, Zhen and Li, Dongsheng and Lv, Yiwei},
  booktitle={Proceedings of ACL},
  year={2019}
}

Smart batching reference and best reference by far:

http://mccormickml.com/2020/07/29/smart-batching-tutorial/#uniform-length-batching
https://colab.research.google.com/drive/1Er23iD96x_SzmRG8md1kVggbmz0su_Q5#scrollTo=qRWT-D4U_Pvx

First steps reference:

#code extracted from tutorial at http://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/


Helper links
https://mccormickml.com/2020/03/10/question-answering-with-a-fine-tuned-BERT/#2-load-fine-tuned-bert-large
https://programmerbackpack.com/bert-nlp-using-distilbert-to-build-a-question-answering-system/
https://keras.io/examples/nlp/text_extraction_with_bert/

'''