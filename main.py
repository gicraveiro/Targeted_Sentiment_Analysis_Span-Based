#import bert.tokenization as tokenizer
import torch
import transformers # pytorch transformers
import pandas
import math
import random
from transformers import AutoConfig, AdamW, get_linear_schedule_with_warmup#BertConfig
#from transformers.utils.dummy_pt_objects import LongformerForQuestionAnswering
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
  #segment_ids = []

  for batch,labels in zip(random_batches_list,random_labels_list):
    max_len = 0
    for sentence in batch:
      padded_batch = []
      batch_attention_mask = []
      batch_start_positions = []
      batch_end_positions = []
      #batch_segment_ids = []

      if (len(sentence) > max_len):
        max_len = len(sentence)
    
    for sentence, sent_label in zip(batch,labels):
      sentence_start_positions = []
      sentence_end_positions = []
      num_zeros = max_len - len(sentence)
      sentence_attention_mask = (len(sentence)*[1] + num_zeros*[0])
      sent_label_list = sent_label.split()
      
      label_i = -1
      finished = 0
      #for index in enumerate(sent_label_list):
      for token_i,tok in enumerate(sentence):
        #print("oi")
        if label_i < 0 or label_i >= len(sent_label_list):
          tag = 'O'    
        else:
          tag = sent_label_list[label_i].split("=")
          tag = tag[1]

        end_index = token_i
        if( tag != 'O'):
          sentence_start_positions = [token_i]
          sentence_end_positions = [token_i]
        while(tag != 'O' and label_i < len(sent_label_list)):
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
      sentence = sentence + [0] * num_zeros
      #batch_segment_ids = max_len * [0]
      padded_batch.append(sentence)
      batch_attention_mask.append(sentence_attention_mask)
      batch_start_positions += sentence_start_positions
      batch_end_positions += sentence_end_positions
    
    input_ids.append(torch.tensor(padded_batch, dtype=torch.long))
    attention_mask.append(torch.tensor(batch_attention_mask, dtype=torch.long))
    start_positions.append(torch.tensor(batch_start_positions, dtype=torch.long))
    end_positions.append(torch.tensor(batch_end_positions, dtype=torch.long))
    #segment_ids.append(torch.tensor(batch_segment_ids, dtype=torch.long))

  return(input_ids,attention_mask, start_positions, end_positions)

config = AutoConfig.from_pretrained(pretrained_model_name_or_path='distilbert-base-uncased')#,num_labels=2)
#config = BertConfig(vocab_size=30522)
#config = BertConfig.from_json_file("bert/bert_config.json") # include bert directory ONLY in local repository

qa_model_class, tokenizer_class, pretrained_weights = (transformers.DistilBertForQuestionAnswering, transformers.DistilBertTokenizer, 'distilbert-base-uncased') 

# Load pretrained model/tokenizer
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = qa_model_class.from_pretrained(pretrained_weights) 



# TRAINING CONFIGURATIONS

epochs_qnt = 3#3 # TO DO: CHANGE TO 3
batch_size = 8
with open('data/laptop14_train.txt') as file:
  train_dataset_len = sum(1 for line in file)
#with open('data/laptop14_test.txt') as file:
#  test_dataset_len = sum(1 for line in file)
#print(train_dataset_len)
#print( test_dataset_len)
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
  input_ids,  attention_mask, start_positions, end_positions = restart_sampling(batch_size, "data/laptop14_train.txt")

  for batch_index,(input_ids, input_mask, input_start, input_end) in enumerate(zip(input_ids, attention_mask, start_positions, end_positions)):

    model.zero_grad() #clear previous gradients

    #loss is returned because it is supervised learning based on the labels
    # logits are the predicted outputs by the model before activation
    outputs = model(input_ids=input_ids,  attention_mask=input_mask, start_positions=input_start, end_positions=input_end)
    loss = outputs.loss
    #start_logits = outputs.start_logits
    #end_logits = outputs.end_logits
    print("Loss", loss)
    #print(start_logits, end_logits)
    
    loss.backward() # backward propagate
    train_loss += loss.item()

    # Clip the norm of the gradients to 1.0.
    # This is to help prevent the "exploding gradients" problem.
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    optimizer.step() #update parameters 
    scheduler.step()
    #model.zero_grad()

print("Average loss", train_loss/math.ceil(train_dataset_len/batch_size))
print("Congrats!Training concluded successfully!")

#EVALUATION

  # PREPARE DATASET ON TEST
  # need to send the right parameters!
input_ids, attention_mask, start_positions, end_positions = restart_sampling(batch_size, "data/laptop14_test.txt")

model.eval() #set to evaluation mode

predicted_starts, predicted_ends, real_starts, real_ends = [], [], [], []
count = 0
real_count, pred_count = 0,0

for batch_index,(input_ids, input_mask, input_start, input_end) in enumerate(zip(input_ids, attention_mask, start_positions, end_positions)):

  with torch.no_grad():
    outputs = model(input_ids, attention_mask=input_mask) 
    
  #logits = outputs[0]
  start_logits = outputs.start_logits
  end_logits = outputs.end_logits
  print(start_logits)
  print(end_logits)
  for logit in start_logits:
    start_logits = logit.numpy()
    end_logits = logit.numpy()
    pred_start = numpy.max(start_logits)
    pred_end = numpy.max(end_logits)
    pred_start = numpy.nonzero(start_logits == pred_start)
    pred_end = numpy.nonzero(end_logits == pred_end)

    predicted_starts.append(pred_start[0][0])
    predicted_ends.append(pred_end[0][0])

  real_count += len(input_start)
  #pred_count += len(predicted_starts)
  real_starts.append(input_start)
  real_ends.append(input_end)

print('Congrats! Evaluation concluded successfully!')
print(predicted_starts)
print(predicted_ends)
#print(len(predicted_starts))
#print(len(predicted_ends))

total_real_starts = numpy.concatenate(real_starts, axis=0)
total_real_ends = numpy.concatenate(real_ends, axis=0)
#total_real_starts = total_real_starts[total_real_starts != -1]
#total_real_ends = total_real_ends[total_real_ends != -1]
total_real_starts[total_real_starts == -1] = 0
total_real_ends[total_real_ends == -1] = 0
total_real_starts = list(total_real_starts)
total_real_ends = list(total_real_ends)


print(total_real_starts, len(total_real_starts))
print(total_real_ends, len(total_real_ends))
print(len(predicted_starts))
print(len(predicted_ends))
print("count", count, real_count, pred_count)
true_positives = 0
for index, (pred_start, real_start, pred_end, real_end) in enumerate(zip(predicted_starts,total_real_starts, predicted_ends, total_real_ends)):
  if(predicted_starts[index] == total_real_starts[index] and predicted_ends[index] == total_real_ends[index]):
    true_positives += 1
print(true_positives)
print("Accuracy", true_positives/len(predicted_starts))
# ADD REPORTS OF SIZE AND TIME !! -- EFFICIENCY

#total_predicted_starts = numpy.concatenate(predicted_starts, axis=0)
#total_predicted_ends = numpy.concatenate(predicted_ends, axis=0)

#predicted_start = numpy.argmax(predicted_starts, axis=1).flatten()
#predicted_end = numpy.argmax(predicted_ends, axis=1).flatten()

#accuracy_starts = (predicted_starts == total_real_starts).mean()
#accuracy_ends = (predicted_ends == total_real_ends).mean()
#true_positives = 

#print('Accuracy Start:', accuracy_starts)
#print('Accuracy Ends', accuracy_ends)


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