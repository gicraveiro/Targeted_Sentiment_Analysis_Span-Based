#import bert.tokenization as tokenizer
import torch
import transformers # pytorch transformers
import pandas
import numpy

#file = open("data/laptop14_test.txt")
#laptop_train = open("data/laptop14_train.txt")

#code extracted from tutorial at http://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/
model_class, tokenizer_class, pretrained_weights = (transformers.DistilBertModel, transformers.DistilBertTokenizer, 'distilbert-base-uncased')
#model_class, tokenizer_class, pretrained_weights = (transformers.BertModel, transformers.BertTokenizer, 'bert-base-uncased')

# Load pretrained model/tokenizer
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights)

dataframe = pandas.read_csv("data/laptop14_train.txt", delimiter='####', header=None, names=['text','labels'],engine='python')

#dataframe tokenized dataset
print(dataframe['text'].str.len().sort_values())
print(dataframe['text'].str.len().sort_values().index)
new_index_list = dataframe['text'].str.len().sort_values().index
dataframe = dataframe.reindex(new_index_list) # sorted dataframe
print(dataframe)
print(dataframe['text'].str.len().sort_values())
dataframe = dataframe.reset_index(drop=True)
print(dataframe)
tokenized_dataset = dataframe['text'].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))#, max_length=100, truncation=True, padding=False''' )

print(tokenized_dataset[0])
print(tokenized_dataset[3044])
#print(tokenized_dataset['text'].str.len().sort_values())
#list format tokenized dataset
#tokenized_dataset = []
#print(.values.len().sort_values())
#for sentence in dataframe[0]:
#  print(sentence)
#  tokenized_sentence = tokenizer.encode(sentence, add_special_tokens=True)
#  tokenized_dataset.append(tokenized_sentence)

#print(tokenized_dataset)
#print(numpy.array(tokenized_dataset).shape)

max_len = 0
for sentence in tokenized_dataset.values:
  if (len(sentence) > max_len):
    max_len = len(sentence)
print(max_len)

#unsorted_lengths = [len(x) for x in tokenized_dataset].sort_values()
#print(type(tokenized_dataset))
#print(tokenized_dataset.)
#tokenized_dataset.reindex(tokenized_dataset.name.len().sort_values().index)
#tokenized_dataset.sort_values(key=len(x))
#sorted_lengths = [len(x) for x in tokenized_dataset]
#print(sorted_lengths)

padded_tok_dataset = numpy.array([i + [0]*(max_len-len(i)) for i in tokenized_dataset.values]) # TO DO: UNDERSTAND
#print(padded_tok_dataset)
print(numpy.array(padded_tok_dataset).shape)

attention_mask = numpy.where(padded_tok_dataset != 0, 1, 0)
#print(attention_mask.shape)

input_ids = torch.tensor(padded_tok_dataset)  
#print(input_ids)
attention_mask = torch.tensor(attention_mask)
#print(attention_mask)
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