import torch
import transformers # pytorch transformers
import pandas
import math
import random
import time
from transformers import AutoConfig, AdamW, DistilBertConfig, get_linear_schedule_with_warmup
import numpy

def initial_messages():
  print("\nHey visitor, welcome!\n")

  print("My name is Giovana, I am the NLP enthusiast behind this experimental implementation") 
  print("This is the implementation of some Natural Language Understanding (NLU) tasks\n")

  print("All this work has been done as the final project of the NLU course at the University of Trento")
  print("The final project proposition chosen was Sentiment Analysis 3 - Target extraction and Polarity CLassification on dataset SemEval2014")
  print("Further orientations were given by the following paper:")

  print("\n'Minghao Hu, Yuxing Peng, Zhen Huang, Dongsheng Li, and Yiwei Lv. 2019b. Open-domain targeted sentiment analysis via span-based extraction and classification. In ACL, pages 537–546.\n")

  print("The authors of this paper implemented a few approaches on target extraction and polarity classification, affirming that best results were given by a pipeline of span multi-target extraction and classification")
  print("Unfortunately, multi-target extraction got too complicated")
  print("Instead, I considered only the first target of each annotated sentence")
  print("Also, I implemented the target extraction and classification separately (so not a pipeline)")

  print("\nFurthermore, when I say I implemented, I meant that I:")
  print(" Searched tutorials to learn how to manipulate pre-trained bert (and variations) models and speed up training")
  print("   1-> Configured it to a seemingly adequate distilBERT model")
  print("   2-> Formatted input performing smart batching")
  print("   3-> Fed the input to bert training with the laptop14_train.txt dataset")
  print("   4-> Fed the input to bert evaluation with the laptop14_test.txt dataset")
  print("   5-> Reported results\n")

  print("For target extraction, the model used was Distilbert Question Answering")
  print("For polarity classification, the model used was Distilbert Sequence Classification")

  print("\nModel choices and configurations were motivated by ethical guidelines that encourage research line using relatively small datasets, other than limited time and power resources.")
  print("All the chosen configurations were also replicated from tutorials, previous research and/or model designer recommendations")
  print("The mix of used configurations, each one extracted possibly from a different source, however, is not guaranteed to provide state-of-the-art results.")
  print("The project elaboration goal was above all to provide me a bit more understanding and practical experience of NLU current concepts, tools, datasets, configurations, etc, to which it has been succesful =D\n")

  print("To perform target extraction, enter 0")
  print("To perform polarity classification, enter 1")
  print("To exit, enter 2")

  print("\nJust so you are prepared") 
  print("Target extraction takes about 17 minutes to finish")
  print("And polarity classification takes about 19 minutes to finish\n")

def read_dataset(input_file, tokenizer):
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

  #print(tokenized_dataframe)
  #print(dataframe)

  return tokenized_dataframe

# RANDOM BATCH REORDERING
def restart_sampling(batch_size, tokenized_dataframe):

  # copying dataframe to manipulate it
  dynamic_dataframe = tokenized_dataframe.copy(deep=True)
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
    random_batches_list.append(batch['text'])
    random_labels_list.append(batch['Labels']) 
    dynamic_dataframe.drop(dynamic_dataframe.index[random_index:random_index+batch_size], inplace=True)

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
          #if (tag != 'NEU'):
          sentence_start_positions = [token_i]
          sentence_end_positions = [token_i]
          if(tag == 'POS'):
            sentence_polarities = [0]
          elif(tag == 'NEG'):
            sentence_polarities = [1]
          elif(tag ==  'NEU'):
            sentence_polarities = [2]
        else:
          sentence_polarities = [3]
          
        while(tag != 'O' and label_i < len(sent_label_list)): # and tag != 'NEU'
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
        sentence_polarities = [3] # 3 EQUALS TO NO TARGETED SPAN IN THIS SEQUENCE, 2 if neutrals have been filtered
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

def precision(true_positives, false_positives):
  if (true_positives+ false_positives <= 0):
    return 0
  precision = true_positives / (true_positives + false_positives)
  return precision

def recall(true_positives, false_negatives):
  if (true_positives+ false_negatives <= 0):
    return 0
  recall = true_positives / (true_positives + false_negatives)
  return recall

def f1(precision, recall):
  if (precision + recall <= 0):
    return 0
  f1 = (2*precision*recall)/(precision+recall)
  return f1

def target_extraction_training(model, optimizer, epochs_qnt, training_steps, batch_size, training_tokenized_dataframe):
  # TRAINING STEP
  model.train() # only sets the training mode
  #num warmup steps is default value in glue.py
  scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0,num_training_steps = training_steps)
 
  # HERE IS WHERE THE TRAINING REALLY STARTS
  # All this steps were replicated from the smart batching tutorial
  total_train_loss = 0.0
  for epoch in range(0,epochs_qnt):
    print("\nEPOCH", epoch, "\n")
    train_loss = 0.0
    # FORMATTING DATA WITH SMART BATCHING
    # At every epoch the data is sorted randomly
    print("\nRANDOMIZING BATCHES\n")
    #print(training_tokenized_dataframe)
    input_ids,  attention_mask, start_positions, end_positions, polarities = restart_sampling(batch_size, training_tokenized_dataframe)
    #print(input_ids)
    print("\nBATCHES HAVE BEEN RANDOMIZED\n")

    for batch_index,(input_ids_batch, input_mask_batch, input_start_batch, input_end_batch) in enumerate(zip(input_ids, attention_mask, start_positions, end_positions)):

      model.zero_grad() #clear previous gradients

      #loss is returned because it is supervised learning based on the labels
      # logits are the predicted outputs by the model before activation  
      outputs = model(input_ids=input_ids_batch,  attention_mask=input_mask_batch, start_positions=input_start_batch, end_positions=input_end_batch)
      loss = outputs.loss
      loss.backward() # backward propagate
      train_loss += loss.item()

      if( batch_index %38  == 0):
        print("TRAINING UPDATE")
        print("CURRENT AVERAGE LOSS:", train_loss/(batch_index+1))

      # Clip the norm of the gradients to 1.0.
      # This is to help prevent the "exploding gradients" problem. 
      torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

      optimizer.step() #update parameters 
      scheduler.step()
      #model.zero_grad()
    
    total_train_loss += train_loss/(len(input_ids))
    print("\nAverage loss after running epoch", epochs_qnt,"is",total_train_loss,"\n")
 
  return total_train_loss/epochs_qnt

def target_extraction_evaluation(model, testing_tokenized_dataframe):
  # PREPARE DATASET ON TEST
  input_ids, attention_mask, start_positions, end_positions, polarities = restart_sampling(batch_size, testing_tokenized_dataframe)

  model.eval() # set to evaluation mode

  predicted_starts, predicted_ends, real_starts, real_ends = [], [], [], []

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
  return real_starts, real_ends, predicted_starts, predicted_ends

def target_extraction_results(real_starts, real_ends, predicted_starts, predicted_ends):
  
  total_real_starts = numpy.concatenate(real_starts, axis=0)
  total_real_ends = numpy.concatenate(real_ends, axis=0)
  #total_real_starts[total_real_starts == -1] = 0
  #total_real_ends[total_real_ends == -1] = 0
  total_real_starts = list(total_real_starts)
  total_real_ends = list(total_real_ends)

  true_positives = 0
  #false_positives = 0
  #false_negatives = 0
  for index, (pred_start, real_start, pred_end, real_end) in enumerate(zip(predicted_starts,total_real_starts, predicted_ends, total_real_ends)):
    if(predicted_starts[index] == total_real_starts[index] and predicted_ends[index] == total_real_ends[index]):
      true_positives += 1

  total_time = time.time() - initial_time
  print("\nTrue positives include only cases in which start and end were predicted correctly, only the first target of the span and cases where the model predicted correctly that there were no targets in the span\n")
  print("Total true positives:", true_positives)
  print("Accuracy", true_positives/(len(predicted_starts))) # TO DO: CORRECT THIS
  print("Total target extraction time in minutes:", round((total_time/60),2))


  total_time = time.time() - initial_time #total_time + training_time + eval_time



 # TO DO: calculate efficiency measure + precision, recall, f1
 # print size measure
 # print time measure
 
def target_extraction(training_tokenized_dataframe, testing_tokenized_dataframe, epochs_qnt, training_steps, batch_size, initial_time, train_dataset_len):
  # Load configurations and pretrained model
  config = AutoConfig.from_pretrained(pretrained_model_name_or_path='distilbert-base-uncased', n_layers=4, hidden_dim=1200, dim=312, max_position_embeddings=512)
  print("Configurations:\n", config)
  qa_model = transformers.DistilBertForQuestionAnswering(config) 
  model = qa_model.from_pretrained(pretrained_weights) 
  
  # Setting optimizer
  # This is the learning rate Hu et al (2019) used # args.adam_epsilon default is 1e-8.
  optimizer = AdamW(model.parameters(),lr = 2e-5,eps = 1e-8  ) # TO DO: change learning rate?

  # TRAINING
  avg_train_loss = target_extraction_training(model, optimizer, epochs_qnt, training_steps, batch_size, training_tokenized_dataframe)
  
  training_time = time.time() - initial_time
  print("Congrats! Training concluded successfully!\n")
  print("Total Average loss", avg_train_loss)# train_loss/math.ceil(train_dataset_len/batch_size))
  print("Training time in minutes:", round((training_time/60),2),"\n")
  
  #EVALUATION
  real_starts, real_ends, predicted_starts, predicted_ends = target_extraction_evaluation(model, testing_tokenized_dataframe)

  eval_time = time.time() - training_time - initial_time
  print('Congrats! Evaluation concluded successfully!\n')
  print('Evaluation time in seconds:', round(eval_time,2))

  target_extraction_results(real_starts, real_ends, predicted_starts, predicted_ends)

def polarity_classification_training(model, optimizer, epochs_qnt, training_steps, batch_size, training_tokenized_dataframe):
  model.train() # only sets the training mode
  #num warmup steps is default value in glue.py
  scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0.1,num_training_steps = training_steps)
  for epoch in range(0,epochs_qnt):
    print("\nEPOCH", epoch, "\n")
    train_loss = 0.0
    print("\nRANDOMIZING BATCHES\n")
    input_ids,  attention_mask, start_positions, end_positions, polarities = restart_sampling(batch_size, training_tokenized_dataframe)
    print("\nBATCHES HAVE BEEN RANDOMIZED\n")

    for batch_index,(input_ids, input_mask, batch_polarities) in enumerate(zip(input_ids, attention_mask, polarities)):

      model.zero_grad() #clear previous gradients
      #print(batch_polarities)
      #loss is returned because it is supervised learning based on the labels
      # logits are the predicted outputs by the model before activation
      outputs = model(input_ids=input_ids,  attention_mask=input_mask, labels=batch_polarities)
      loss = outputs.loss
      
      loss.backward() # backward propagate
      train_loss += loss.item()

      if( batch_index %80  == 0):
        print("TRAINING UPDATE")
        print("CURRENT AVERAGE LOSS:", train_loss/(batch_index+1))

      # Clip the norm of the gradients to 1.0.
      # This is to help prevent the "exploding gradients" problem.
      torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

      optimizer.step() #update parameters 
      scheduler.step()
  return train_loss

def polarity_classification_evaluation(model, batch_size, tokenized_testing_dataset):
  # PREPARE DATASET ON TEST
  input_ids, attention_mask, start_positions, end_positions, polarities = restart_sampling(batch_size, tokenized_testing_dataset)

  model.eval() # set to evaluation mode

  predicted_polarities, real_polarities = [], []

  for batch_index,(input_ids, input_mask, input_start, input_end, polarities_batch) in enumerate(zip(input_ids, attention_mask, start_positions, end_positions, polarities)):

    with torch.no_grad():
      outputs = model(input_ids, attention_mask=input_mask, labels=polarities_batch) 

    polarity_logits = outputs.logits
    #print(polarity_logits)

    for logit in polarity_logits:
      polarity_logits = logit.numpy()
      pred_pol = numpy.max(polarity_logits)
      pred_pol = numpy.nonzero(polarity_logits == pred_pol)
      predicted_polarities.append(pred_pol[0][0])

    real_polarities.append(polarities_batch)
  return real_polarities, predicted_polarities

def polarity_classification_results(real_polarities, predicted_polarities):
  total_real_polarities = numpy.concatenate(real_polarities, axis=0)
  unique, frequency = numpy.unique(total_real_polarities,return_counts = True) 
  #print(unique, frequency)
  #total_real_polarities[total_real_polarities == 2] = 0
  total_real_polarities = list(total_real_polarities)

  true_positives_total = 0
  true_positives_POS = 0
  true_positives_NEG = 0
  true_positives_NEU = 0
  true_positives_ABSENT = 0

  false_positives_total = 0
  false_positives_POS = 0
  false_positives_NEG = 0
  false_positives_NEU = 0
  false_positives_ABSENT = 0

  false_negatives_total = 0
  false_negatives_POS = 0
  false_negatives_NEG = 0
  false_negatives_NEU = 0
  false_negatives_ABSENT = 0

  for index, (pred_pol, real_pol) in enumerate(zip(predicted_polarities,total_real_polarities)):
    pred_aux = predicted_polarities[index]
    real_aux = total_real_polarities[index]
    if(pred_aux == real_aux):
      #print("correct prediction", pred_aux, real_aux)
      true_positives_total += 1
      if(pred_aux == 0):
        true_positives_POS += 1
      elif(pred_aux == 1):
        true_positives_NEG += 1
      elif(pred_aux == 2):
        true_positives_NEU += 1
      elif(pred_aux == 3):
        true_positives_ABSENT += 1
        true_positives_total -= 1

    else:
      false_positives_total += 1
      false_negatives_total += 1
      #print("WRONG", pred_aux, real_aux)
      if(pred_aux == 0):
        false_positives_POS += 1
      elif(pred_aux == 1):
        false_positives_NEG += 1
      elif(pred_aux == 2): 
        false_positives_NEU += 1
      elif(pred_aux == 3):
        false_positives_ABSENT += 1
        false_negatives_total -= 1
      
      if(real_aux == 0):
        false_negatives_POS += 1
      elif(real_aux == 1):
        #print("negative")
        false_negatives_NEG += 1
      elif(real_aux == 2):
        false_negatives_NEU += 1
      elif(real_aux == 3):
        false_negatives_ABSENT += 1
        #false_positives_total -= 1

  precision_total = precision(true_positives_total,false_positives_total)
  recall_total = recall(true_positives_total,false_negatives_total)
  f1_total = f1(precision_total,recall_total)

  precision_POS = precision(true_positives_POS, false_positives_POS)
  recall_POS = recall(true_positives_POS, false_negatives_POS)
  f1_POS = f1(precision_POS, recall_POS)

  precision_NEG = precision(true_positives_NEG, false_positives_NEG)
  recall_NEG = recall(true_positives_NEG, false_negatives_NEG)
  f1_NEG = f1(precision_NEG, recall_NEG)

  precision_NEU = precision(true_positives_NEU, false_positives_NEU)
  recall_NEU = recall(true_positives_NEU, false_negatives_NEU)
  f1_NEU = f1(precision_NEU, recall_NEU)

  precision_ABSENT = precision(true_positives_ABSENT, false_positives_ABSENT)
  recall_ABSENT = recall(true_positives_ABSENT, false_negatives_ABSENT)
  f1_ABSENT = f1(precision_ABSENT, recall_ABSENT)

  total_time = time.time() - initial_time #total_time + training_time + eval_time
  print("Total true positives", true_positives_total)
  print("True Positives Positive:", true_positives_POS, "True Positives Negative:", true_positives_NEG, "True Positives Neutral:", true_positives_NEU, "True Positives Absent:", true_positives_ABSENT)
  print("False Positives Positive:", false_positives_POS, "False Positives Negative:", false_positives_NEG, "False Positives Neutral:", false_positives_NEU, "False Positives Absent:", false_positives_ABSENT)
  print("False Negatives Positive:", false_negatives_POS, "False Negatives Negative:", false_negatives_NEG, "False Negatives Neutral:", false_negatives_NEU, "False Negatives Absent:", false_negatives_ABSENT)
  print("\nAccuracy", true_positives_total/(len(predicted_polarities)-true_positives_ABSENT-false_negatives_ABSENT))
  print("Total time in minutes:", round((total_time/60),2))

  print("\nPrecisions:")
  print("Positive", precision_POS)
  print("Negative", precision_NEG)
  print("Neutral", precision_NEU)
  print("Absent", precision_ABSENT)
  print("Total", precision_total)

  print("\nRecalls:")
  print("Positive", recall_POS)
  print("Negative", recall_NEG)
  print("Neutral", recall_NEU)
  print("Absent", recall_ABSENT)
  print("Total", recall_total)

  print("\nF1 Scores")
  print("Positive", f1_POS)
  print("Negative", f1_NEG)
  print("Neutral", f1_NEU)
  print("Absent", f1_ABSENT)
  print("Total", f1_total)
  # TO DO: report in a more organized way the EFFICIENCY based on size and time

def polarity_classification(training_tokenized_dataframe, tokenized_testing_dataset, epochs_qnt, training_steps, batch_size):
  # labels = Positive, Negative, -removed Neutral for now- , No target in the span
  config = DistilBertConfig(pretrained_model_name_or_path='distilbert-base-uncased',  num_labels=4) #, n_layers=4, hidden_dim=1200, dim=312)#, max_position_embeddings=312) 
  model = transformers.DistilBertForSequenceClassification(config)
  print("Configurations:\n", config)

  # Setting optimizer
  # This is the learning rate the paper used # args.adam_epsilon  - default is 1e-8.
  optimizer = AdamW(model.parameters(),lr = 2e-5,eps = 1e-8  ) #change learning rate

  #scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0,num_training_steps = training_steps)
  
  # TRAINING STEP
  train_loss = polarity_classification_training(model, optimizer, epochs_qnt, training_steps, batch_size, training_tokenized_dataframe)

  class_training_time = time.time() - initial_time #-total_time
  print("\n\n\nCongrats! Classification Training concluded successfully!\n")
  print("Average loss", train_loss/math.ceil(train_dataset_len/batch_size))
  print("Classification training time in minutes:", round((class_training_time/60),2))

  real_polarities, predicted_polarities = polarity_classification_evaluation(model, batch_size, tokenized_testing_dataset)

  eval_time = time.time() - class_training_time #- total_time
  print('Congrats! Evaluation concluded successfully!\n')
  print('Classification evaluation time in seconds:', round(eval_time,2))

  polarity_classification_results(real_polarities, predicted_polarities)
  
# MAIN

# START OF THE CODE

initial_messages()

opt = -1
while (opt != 0 and opt != 1 and opt != 2):
  opt = int(input())

if(opt == 2):
  quit()

# CRONOMETER START
initial_time, format_time = time.time(), time.asctime( time.localtime(time.time()) )
print("\nAllright, let's get started! The cronometer starts ticking now")
print(format_time, "\n")

#  COMMON CONFIGURATIONS

with open('data/laptop14_train.txt') as file:
  train_dataset_len = sum(1 for line in file)
#with open('data/laptop14_train.txt') as file:
#  test_dataset_len = sum(1 for line in file)

epochs_qnt = 3
batch_size = 8
training_steps = epochs_qnt * math.ceil(train_dataset_len/batch_size)
print("ADDITIONAL CONFIGURATIONS\n")
print("\nNUMBER OF EPOCHS:", epochs_qnt)
print("BATCH SIZE:", batch_size)
print("TRAINING STEPS:", training_steps)

# Load tokenizer
tokenizer_class, pretrained_weights = (transformers.DistilBertTokenizer, 'distilbert-base-uncased')
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)

# Read, reorder and tokenize training and testing datasets
print("\nREORGANIZING DATASET TO PERFORM SMART BATCHING\n")
training_tokenized_dataframe = read_dataset('data/laptop14_train.txt', tokenizer)
testing_tokenized_dataframe  = read_dataset('data/laptop14_test.txt', tokenizer)

if(opt == 0):
  print("\nTARGET EXTRACTION STARTING\n")
  target_extraction(training_tokenized_dataframe, testing_tokenized_dataframe, epochs_qnt, training_steps, batch_size, initial_time, train_dataset_len)
elif(opt == 1):
  print("\nPOLARITY CLASSIFICATION STARTING\n")
  polarity_classification(training_tokenized_dataframe, testing_tokenized_dataframe, epochs_qnt, training_steps, batch_size)

total_time = time.time() - initial_time
print("\nWe're done! Total time was", round(total_time/60,2), "minutes") # or", total_time, "seconds")
print("\nThank you for the patience, come back anytime!\n")


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

tutorial at http://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/


Helper links
https://mccormickml.com/2020/03/10/question-answering-with-a-fine-tuned-BERT/#2-load-fine-tuned-bert-large
https://programmerbackpack.com/bert-nlp-using-distilbert-to-build-a-question-answering-system/
https://keras.io/examples/nlp/text_extraction_with_bert/

'''