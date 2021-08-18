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

# TOKENIZATION
tokenized_dataset = dataframe['text'].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))#, max_length=100, truncation=True, padding=False''' )

# RANDOM BATCH REORDERING

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

  dynamic_dataframe.drop(dynamic_dataframe.index[random_index:random_index+batch_size], inplace=True)

# PADDING AND ATTENTION MASK WITH SMART BATCHING

attention_mask = []
input_ids = []

for batch in random_batches_list:
  max_len = 0
  for sentence in batch:
    padded_batch = []
    batch_attention_mask = []

    if (len(sentence) > max_len):
      max_len = len(sentence)
  
  for sentence in batch:
    num_zeros = max_len - len(sentence)
    sentence_attention_mask = (len(sentence)*[1] + num_zeros*[0])
    sentence = sentence + [0] * num_zeros
    padded_batch.append(sentence)
    batch_attention_mask.append(sentence_attention_mask)

  input_ids.append(torch.tensor(padded_batch))
  attention_mask.append(torch.tensor(batch_attention_mask))


# Get the new list of lengths after sorting.
padded_lengths = []

# For each batch...
for batch in input_ids:
    
    # For each sample...
    for s in batch:
    
        # Record its length.
        padded_lengths.append(len(s))

# Sum up the lengths to the get the total number of tokens after smart batching.
smart_token_count = numpy.sum(padded_lengths)

# To get the total number of tokens in the dataset using fixed padding, it's
# as simple as the number of samples times our `max_len` parameter (that we
# would pad everything to).
fixed_token_count = 91*3045

# Calculate the percentage reduction.
prcnt_reduced = (fixed_token_count - smart_token_count) / float(fixed_token_count) 

print('Total tokens:')
print('   Fixed Padding: {:,}'.format(fixed_token_count))
print('  Smart Batching: {:,}  ({:.1%} less)'.format(smart_token_count, prcnt_reduced))


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