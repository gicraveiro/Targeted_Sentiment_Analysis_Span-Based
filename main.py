#import bert.tokenization as tokenizer
import torch
import transformers # pytorch transformers
import pandas
import numpy
import random
from reused import BertConfig, BertForSpanAspectExtraction, run_train_epoch, read_eval_data, read_train_data, prepare_optimizer




def restart_sampling():
  # TOKENIZATION
  tokenized_dataset = dataframe['text'].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))#, max_length=100, truncation=True, padding=False''' )
  labels_list = dataframe['labels'].to_list()
  tokenized_dataframe = tokenized_dataset.to_frame()
  tokenized_dataframe.insert(1, "Labels", labels_list, True)

  # RANDOM BATCH REORDERING

  batch_size = 8
  dynamic_dataframe = tokenized_dataset.copy(deep=True) #copy of the sentences column of the dataset to delete it parts
  dynamic_labels = dataframe['labels']
  #print(dynamic_labels, type(dynamic_labels))
  #dynamic_labels = tokenized_dataframe['labels'].copy(deep=True)
  #print(dynamic_labels)
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

    #print(batch, batch_labels)
  #print(dynamic_dataframe)
  #print(dynamic_labels)

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
      sentence_start_positions = [0]
      sentence_end_positions = [0]
      print(sentence_start_positions, sentence_end_positions)
      num_zeros = max_len - len(sentence)
      sentence_attention_mask = (len(sentence)*[1] + num_zeros*[0])
      sent_label_list = sent_label.split()
      print(sent_label_list)
      
      for index,token in enumerate(sent_label_list):
        tag = token.split("=")
        tag = tag[1]
        #print(sentence[index], ' ', tokenizer.convert_ids_to_tokens(sentence[index]), tag)
        print(sentence_start_positions)
        print(sentence_end_positions)
        print(tokenizer.convert_ids_to_tokens(sentence[index])[0:2])
        if (tokenizer.convert_ids_to_tokens(sentence[index])[0:2] == '##'):
          print(sentence_start_positions[-1])
          print(sentence_start_positions)
          print(tag)
          sentence_start_positions += [sentence_start_positions[-1]]
          sentence_end_positions += [sentence_end_positions[-1]]
          print("added")

        if (tag != 'O' and (index-1 < 0 or sent_label_list[index-1].split("=")[1] == 'O')):
          sentence_start_positions += [1]
        else:
          sentence_start_positions += [0]
        
        if (tag != 'O' and (index+1 > len(sent_label_list) or sent_label_list[index+1].split("=")[1] == 'O')):
          sentence_end_positions += [1]
        else:
          sentence_end_positions += [0]
        print(sentence[index], tokenizer.convert_ids_to_tokens(sentence[index]), sentence_start_positions[index], sentence_end_positions[index], tag)
      
      sentence_start_positions += (num_zeros*[0])+[0] # initial and final token must be added as extra zeros eve beyond the zeros that represent absence of tokens
      sentence_end_positions += (num_zeros*[0])+[0]
      sentence = sentence + [0] * num_zeros
      print(sentence, attention_mask, sentence_start_positions, sentence_end_positions)
      padded_batch.append(sentence)
      batch_attention_mask.append(sentence_attention_mask)
      batch_start_positions.append(sentence_start_positions)
      batch_end_positions.append(sentence_end_positions)

    
    input_ids.append(torch.tensor(padded_batch))
    attention_mask.append(torch.tensor(batch_attention_mask))
    start_positions.append(torch.tensor(batch_start_positions))
    end_positions.append(torch.tensor(batch_end_positions))

  return(input_ids,attention_mask, start_positions, end_positions)

model_class, tokenizer_class, pretrained_weights = (transformers.DistilBertModel, transformers.DistilBertTokenizer, 'distilbert-base-uncased')
#model_class, tokenizer_class, pretrained_weights = (transformers.BertModel, transformers.BertTokenizer, 'bert-base-uncased')

# Load pretrained model/tokenizer
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
#model = model_class.from_pretrained(pretrained_weights) # TO DO: FIGURE OUT THE RIGHT ONE HERE

# Creates a table separating sentences from associated token tags
dataframe = pandas.read_csv("data/laptop14_train.txt", delimiter='####', header=None, names=['text','labels'],engine='python')
#print(dataframe['labels'][0])
tokenized_dataset = dataframe['text'].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))
#print(tokenized_dataset[0])
# Sorts table and transforms each word to the code of the token

new_index_list = dataframe['text'].str.len().sort_values().index
dataframe = dataframe.reindex(new_index_list) # sorted dataframe by length of the sentence
dataframe = dataframe.reset_index(drop=True)

input_ids, attention_mask, start_positions, end_positions = restart_sampling()


# REUSED FROM PAPER 

bert_config = BertConfig.from_json_file("bert/bert_config.json") # include bert directory ONLY in local repository

# tokenization as included in the paper code...
#tokenizer = tokenizer.FullTokenizer(vocab_file="bert/vocab_file.txt", do_lower_case=True)
model = BertForSpanAspectExtraction(bert_config)
# verificação de caminho válido era aqui
#model = bert_load_state_dict(model, torch.load("bert/pytorch_model.bin", map_location='cpu')) # TO DO: EXTRACT FUNCTION FROM PAPER CODE
#print("Loading model from pretrained checkpoint: {}".format("bert/pytorch_model.bin"))
device = "cpu"
model.to(device)

print("***** Preparing data *****")
train_dataloader, num_train_steps = None, None
eval_examples, eval_features, eval_dataloader = None, None, None
train_batch_size = 4 

# their tokenizer was the FullTokenizer but let's head it with this one for now and see what happens
print("***** Preparing training *****")
train_dataloader, num_train_steps = read_train_data(tokenizer)
'''
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

Smart batching reference:

http://mccormickml.com/2020/07/29/smart-batching-tutorial/#uniform-length-batching

First steps reference:

#code extracted from tutorial at http://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/
''' 