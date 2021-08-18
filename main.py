#import bert.tokenization as tokenizer
import torch
import transformers # pytorch transformers
import pandas
import numpy
import random

#code extracted from tutorial at http://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/
model_class, tokenizer_class, pretrained_weights = (transformers.DistilBertModel, transformers.DistilBertTokenizer, 'distilbert-base-uncased')
#model_class, tokenizer_class, pretrained_weights = (transformers.BertModel, transformers.BertTokenizer, 'bert-base-uncased')

# Load pretrained model/tokenizer
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights) # TO DO: FIGURE OUT THE RIGHT ONE HERE

# Creates a table separating sentences from associated token tags
dataframe = pandas.read_csv("data/laptop14_train.txt", delimiter='####', header=None, names=['text','labels'],engine='python')

# Sorts table and transforms each word to the code of the token

new_index_list = dataframe['text'].str.len().sort_values().index
dataframe = dataframe.reindex(new_index_list) # sorted dataframe by length of the sentence
dataframe = dataframe.reset_index(drop=True)
# tokenization
tokenized_dataset = dataframe['text'].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))#, max_length=100, truncation=True, padding=False''' )

#list format tokenized dataset - alternative way
#tokenized_dataset = []
#print(.values.len().sort_values())
#for sentence in dataframe[0]:
#  print(sentence)
#  tokenized_sentence = tokenizer.encode(sentence, add_special_tokens=True)
#  tokenized_dataset.append(tokenized_sentence)

# DEBUG PRINTS
#print(tokenized_dataset)
#print(numpy.array(tokenized_dataset).shape)

batch_size = 8
dynamic_dataframe = tokenized_dataset.copy(deep=True) #copy of the dataframe to delete it parts
random_batches_list = []

while len(dynamic_dataframe) != 0:
  random_index = random.randint(0, len(dynamic_dataframe))
  if (random_index + batch_size >= len(dynamic_dataframe)):
    random_index = random_index - (random_index + batch_size - len(dynamic_dataframe) +1)
    if( random_index < 0):
      random_index = 0
      batch_size = len(dynamic_dataframe)

  batch = dynamic_dataframe[random_index:(random_index+batch_size)]
  random_batches_list.append(batch)
  #for sentence in batch:
  #  random_batches_list.append(sentence)

  dynamic_dataframe.drop(dynamic_dataframe.index[random_index:random_index+batch_size], inplace=True)
  
#print(dynamic_dataframe)
#print(random_batches_list)

# PADDING AND ATTENTION MASK WITH SMART BATCHING

#padded_tok_dataset = []
attention_mask = []
#pytorch_input_ids = []
input_ids = []
#pytorch_attention_mask = []

for batch in random_batches_list:
  max_len = 0
  for sentence in batch:
    padded_batch = []
    batch_attention_mask = []

    if (len(sentence) > max_len):
      max_len = len(sentence)
      #print(max_len)
  
  for sentence in batch:
    num_zeros = max_len - len(sentence)
 #   print(num_zeros, len(sentence), len(sentence) +num_zeros)
    
    #print(sentence)
    sentence_attention_mask = (len(sentence)*[1] + num_zeros*[0])
    sentence = sentence + [0] * num_zeros
    #print(sentence_attention_mask)
    padded_batch.append(sentence)
    batch_attention_mask.append(sentence_attention_mask)

  #padded_tok_dataset.append(padded_batch)
  #attention_mask.append(batch_attention_mask)
#  print(batch_attention_mask,'\n',len(batch_attention_mask))
#  print(padded_batch,'\n',len(padded_batch))
#  input_ids.append(torch.tensor(padded_batch))
#  attention_mask.append(torch.tensor(batch_attention_mask))


#print(padded)
#print(attention_mask)


# PADDING AND ATTENTION MASK WITHOUT SMART BATCHING

#max_len = 0
#for sentence in tokenized_dataset.values:
#  if (len(sentence) > max_len):
#    max_len = len(sentence)
#print(max_len)

#padded_tok_dataset = numpy.array([i + [0]*(max_len-len(i)) for i in tokenized_dataset.values]) # TO DO: UNDERSTAND
#print(padded_tok_dataset)
#print(numpy.array(padded_tok_dataset).shape)

#attention_mask = numpy.where(padded_tok_dataset != 0, 1, 0)
#print(attention_mask.shape)

#input_ids = torch.tensor(padded_tok_dataset)  
input_ids = torch.tensor(input_ids)  
#print(input_ids)
attention_mask = torch.tensor(attention_mask)
#print(attention_mask)

# TRAINING PART THAT MUST BE UNDERSTOOD AND CORRECTED

#with torch.no_grad():
#  last_hidden_states = model(input_ids, attention_mask=attention_mask)
#print(last_hidden_states)
#features = last_hidden_states[0][:,0,:].numpy()

'''
Acknowledgments

Implementation of the architecture proposed by:

@inproceedings{hu2019open,
  title={Open-Domain Targeted Sentiment Analysis via Span-Based Extraction and Classification},
  author={Hu, Minghao and Peng, Yuxing and Huang, Zhen and Li, Dongsheng and Lv, Yiwei},
  booktitle={Proceedings of ACL},
  year={2019}
}
''' 