# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pickle, joblib
import random, time, os, re
from scipy import interp
import xml.etree.ElementTree as et
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix,f1_score,accuracy_score,roc_curve,auc,roc_auc_score
import torch
import torchtext
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, Dataset
from torchtext.vocab import GloVe
from torchtext.legacy.data import Field, BucketIterator, Dataset, Example, TabularDataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import spacy

from google.colab import drive
drive.mount('/content/drive')

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download("punkt")
nlp = spacy.load("en")

if torch.cuda.is_available():
  device = torch.device("cuda")
  print(f"There are {torch.cuda.device_count()} GPUs available!!")
  print("GPU is:", torch.cuda.get_device_name(0))
else:
  print("No GPU available!!")
  device = torch.device("cpu")

# embedding = GloVe(name="840B", dim=300)
# pickle.dump(embedding,open('/content/drive/MyDrive/DL/glove_embedding.pkl','wb'))

embedding = pickle.load(open("/content/drive/MyDrive/DL/glove_embedding.pkl",'rb'))

def convert_xml_to_dataframe(path=""):
  xtree = et.parse(path)
  xroot = xtree.getroot()
  rows = []
  for node in xroot:
    id = node.attrib.get('id')
    txt = node.find("text").text
    asptTerm = node.find("aspectTerms")
    if asptTerm is None:
      continue
    for aspect in node.iter('aspectTerms'):
      for aspt in aspect.iter('aspectTerm'):
        term = aspt.attrib.get('term')
        polarity = aspt.attrib.get('polarity')
        if polarity == 'conflict':
          continue
        rows.append({"context":txt,"target":term,"polarity":polarity})
  df = pd.DataFrame(rows,columns=['context','target','polarity'])
  return df

def encode_labels(df,col_name):
  label = df[col_name]
  l = LabelEncoder()
  label=l.fit_transform(label)
  df[col_name] = label
  return df

laptop_train = convert_xml_to_dataframe('/content/drive/MyDrive/DL/ABSA complete Dataset/ABSA Train/Laptops_Train.xml')
laptop_test = convert_xml_to_dataframe('/content/drive/MyDrive/DL/ABSA complete Dataset/ABSA Test/Laptops_Test_Gold.xml')
restaurant_train = convert_xml_to_dataframe('/content/drive/MyDrive/DL/ABSA complete Dataset/ABSA Train/Restaurants_Train.xml')
restaurant_test = convert_xml_to_dataframe('/content/drive/MyDrive/DL/ABSA complete Dataset/ABSA Test/Restaurants_Test_Gold.xml')
laptop_train = encode_labels(laptop_train,'polarity')
restaurant_train = encode_labels(restaurant_train,'polarity')
laptop_test = encode_labels(laptop_test,'polarity')
restaurant_test = encode_labels(restaurant_test,'polarity')

def tokenize_and_build_vocabulary(context1,target1, context2, target2):
  max_len_context, max_len_target = 0, 0
  tokenized_context1, tokenized_target1 = list(), list()
  tokenized_context2, tokenized_target2 = list(), list()
  target_lens1, context_lens1, target_lens2, context_lens2 = list(), list(), list(), list()
  word2idx = dict()
  word2idx["<pad>"] = 0
  word2idx["<unk>"] = 1
  idx = 2
  for i in range(len(context1)):
    sentence = context1[i].strip()
    # sentence = re.sub('[^a-zA-Z0-9]',' ',sentence)
    sentence = sentence.lower()
    tokenized_sentence1 = word_tokenize(sentence)
    tokenized_context1.append(tokenized_sentence1)
    context_lens1.append(len(tokenized_sentence1)-1)
    for word in tokenized_sentence1:
      if word not in word2idx:
        word2idx[word] = idx
        idx += 1
    max_len_context = max(max_len_context,len(tokenized_sentence1))
    sent = target1[i].strip()
    # sent = re.sub('[^a-zA-Z0-9]',' ',sent)
    sent = sent.lower()
    tokenized_sent1 = word_tokenize(sent)
    tokenized_target1.append(tokenized_sent1)
    target_lens1.append(len(tokenized_sent1))
    for word in tokenized_sent1:
      if word not in word2idx:
        word2idx[word] = idx
        idx += 1
    max_len_target = max(max_len_target,len(tokenized_sent1))
  
  for i in range(len(context2)):
    sentence = context2[i].strip()
    # sentence = re.sub('[^a-zA-Z0-9]',' ',sentence)
    sentence = sentence.lower()
    tokenized_sentence2 = word_tokenize(sentence)
    tokenized_context2.append(tokenized_sentence2)
    context_lens2.append(len(tokenized_sentence2)-1)
    for word in tokenized_sentence2:
      if word not in word2idx:
        word2idx[word] = idx
        idx += 1
    max_len_context = max(max_len_context,len(tokenized_sentence2))
    sent = target2[i].strip()
    # sent = re.sub('[^a-zA-Z0-9]',' ',sent)
    sent = sent.lower()
    tokenized_sent2 = word_tokenize(sent)
    tokenized_target2.append(tokenized_sent2)
    target_lens2.append(len(tokenized_sent2))
    for word in tokenized_sent2:
      if word not in word2idx:
        word2idx[word] = idx
        idx += 1
    max_len_target = max(max_len_target,len(tokenized_sent2))

  return tokenized_context1, tokenized_target1, np.array(context_lens1), np.array(target_lens1), tokenized_context2, tokenized_target2, np.array(context_lens2), np.array(target_lens2), word2idx, max_len_context, max_len_target

def encode_tokenized_text(tokenized_text,word2idx,max_len):
  encoded_text = list()
  for tokenized_sentence in tokenized_text:
    tokenized_sentence += ["<pad>"] * (max_len-len(tokenized_sentence))
    encode_text = [word2idx[word] for word in tokenized_sentence]
    encoded_text.append(encode_text)
  return np.array(encoded_text)

def load_pretrained_vectors(word2idx,embedding,dim):
  cnt = 0
  embedding_matrix = np.random.uniform(-0.1,0.1,(len(word2idx),dim))
  embedding_matrix[word2idx["<pad>"]] = torch.from_numpy(np.zeros((dim,)))
  for word in word2idx.keys():
    if word in embedding.itos:
      cnt += 1
      embedding_matrix[word2idx[word]] = embedding[word]
  print("There are {}/{} pretrained vectors found".format(cnt,len(word2idx)))
  return embedding_matrix

class ABSADataset(Dataset):
  def __init__(self,context,target,polarity,context_lens,target_lens,max_len_context,max_len_target):
    self.target = torch.tensor(target).long()
    self.context = torch.tensor(context).long()
    self.polarity = torch.tensor(polarity).long()
    self.context_lens = torch.tensor(context_lens).long()
    self.target_lens = torch.tensor(target_lens).long()
    self.len = self.polarity.shape[0]
    self.context_mask = torch.zeros(max_len_context, max_len_context)
    self.target_mask = torch.zeros(max_len_target, max_len_target)
    for i in range(max_len_context):
      self.context_mask[i,0:(i+1)] = 1
    for i in range(max_len_target):
      self.target_mask[i,0:(i+1)] = 1

  def __getitem__(self,index):
    return self.context[index], self.target[index], self.polarity[index], self.context_mask[self.context_lens[index]-1],self.target_mask[self.target_lens[index]-1] 

  def __len__(self):
    return self.len

class Attention(nn.Module):
  def __init__(self,query_size,key_size):
    super(Attention,self).__init__()
    self.weights = nn.Parameter(torch.rand(key_size,query_size)*0.2-0.1)
    self.bias = nn.Parameter(torch.zeros(1))

  def forward(self,query,key,mask):
    batch_size = key.size(0)
    time_step = key.size(1)
    weight = self.weights.repeat(batch_size,1,1)
    query = query.unsqueeze(-1)
    mids = weight.matmul(query)
    mids = mids.repeat(time_step,1,1,1).transpose(0,1)
    key = key.unsqueeze(-2)
    scores = torch.tanh(key.matmul(mids).squeeze()+self.bias)
    scores = scores.squeeze()
    scores = scores - scores.max(dim=1, keepdim=True)[0]
    scores = torch.exp(scores)*mask
    attn_weights = scores/scores.sum(dim=1,keepdim=True)
    return attn_weights

class InteractiveAttentionNetwork(nn.Module):

  target_atn = None
  context_atn = None

  def __init__(self,pretrained_embedding=None,freeze_embedding=False,vocab_size=None,
               embed_dim=300,hidden_dim = 300,num_classes = 3,dropout = 0.5,
               max_len_context=0,max_len_target=0,l2_reg=0,attention_flag=True):
    
    super(InteractiveAttentionNetwork,self).__init__()

    self.hidden_dim=hidden_dim
    self.num_classes = num_classes
    self.l2_reg = l2_reg
    self.max_len_context = max_len_context
    self.max_len_target = max_len_target
    self.attention_flag = attention_flag
    # self.target_atn = None
    # self.context_atn = None

    
    if pretrained_embedding is not None:
      self.vocab_size, self.embed_dim = pretrained_embedding.shape
      self.embedding = nn.Embedding.from_pretrained(pretrained_embedding,freeze = freeze_embedding)
    else:
      self.embed_dim = embed_dim
      self.vocab_size = vocab_size
      self.embedding = nn.Embedding(num_embeddings=self.vocab_size,embedding_dim=self.embed_dim,padding_idx=0,max_norm=5.0)
    
    self.target_lstm = nn.LSTM(input_size = self.embed_dim, hidden_size = self.hidden_dim, batch_first = True)
    self.context_lstm = nn.LSTM(input_size = self.embed_dim, hidden_size = self.hidden_dim, batch_first = True)
    if self.attention_flag == True:
      self.target_attn = Attention(self.hidden_dim, self.hidden_dim)
      self.context_attn = Attention(self.hidden_dim, self.hidden_dim)
    self.dropout = nn.Dropout(p=dropout)
    self.fc = nn.Linear(self.hidden_dim*2, self.num_classes)

  def forward(self, context, target, context_mask, target_mask):
    target = self.embedding(target).float()
    target = self.dropout(target)
    target_output, _ = self.target_lstm(target)
    target_output = target_output * target_mask.unsqueeze(-1)
    target_avg = target_output.sum(dim=1, keepdim=False) / target_mask.sum(dim=1, keepdim=True)

    context = self.embedding(context).float()
    context = self.dropout(context)
    context_output, _ = self.context_lstm(context)
    context_output = context_output * context_mask.unsqueeze(-1)
    context_avg = context_output.sum(dim=1, keepdim=False) / context_mask.sum(dim=1, keepdim=True)

    if self.attention_flag == True:
      self.target_atn = self.target_attn(context_avg, target_output, target_mask).unsqueeze(1)
      target_features = self.target_atn.matmul(target_output).squeeze()
      self.context_atn = self.context_attn(target_avg, context_output, context_mask).unsqueeze(1)
      context_features = self.context_atn.matmul(context_output).squeeze()
      features = torch.cat([target_features,context_features], dim=1)
    else:
      features = torch.cat([target_avg, context_avg], dim=1)
    features = self.dropout(features)
    logits = self.fc(features)
    logits = torch.tanh(logits)
    return logits

def initialize_model(pretrained_embedding=None,freeze_embedding=False,vocab_size=None,
                      embed_dim=300,hidden_dim = 300,num_classes = 3,dropout = 0.5,
                      max_len_context=0,max_len_target=0,l2_reg=0,attention_flag=True,learning_rate=0.01):
  ian_model = InteractiveAttentionNetwork(pretrained_embedding=pretrained_embedding, freeze_embedding=freeze_embedding, vocab_size=None, embed_dim=embed_dim,
                  hidden_dim=hidden_dim,num_classes=num_classes, dropout=dropout, max_len_context=max_len_context,
                  max_len_target=max_len_target, l2_reg=l2_reg, attention_flag=attention_flag)
  ian_model.to(device)
  optimizer = optim.Adam(ian_model.parameters(), lr = learning_rate, weight_decay=l2_reg)
  return ian_model, optimizer

def evaluate(model,test_dataloader):
  model.eval()
  test_accuracy, test_loss = list(), list()
  for batch in test_dataloader:
    context, target, labels, context_mask, target_mask = tuple(t.to(device) for t in batch)
    logits = model(context,target,context_mask,target_mask)
    loss = loss_fn(logits,labels)
    preds = torch.argmax(logits,dim=1).flatten()
    test_loss.append(loss.item())
    accuracy = (preds == labels).cpu().numpy().mean() * 100
    test_accuracy.append(accuracy)

  test_loss = np.mean(test_loss)
  test_accuracy = np.mean(test_accuracy)
  return test_loss, test_accuracy

def train(model, optimizer, train_dataloader, test_dataloader=None, epochs=10, path=""):
  best_accuracy = 0
  print(f"{'Epoch':^6} | {'Train Loss':^12} | {'Train Accuracy':^10} | {'Test Loss':^10} | {'Test Accuracy':^10} | {'Elapsed':^9}")
  print("-"*80)
  training_loss, testing_loss = list(), list()
  # loss1, accuracy1 = list(), list()
  for i in range(epochs):
    t0 = time.time()
    loss1, accuracy1 = list(), list()
    total_loss, total_accuracy = 0, 0
    model.train()
    for step,batch in enumerate(train_dataloader):
      context, target, labels, context_mask, target_mask = tuple(t.to(device) for t in batch)
      optimizer.zero_grad()
      logits = model(context,target,context_mask,target_mask)
      loss = loss_fn(logits,labels)
      total_loss += loss.item()
      loss1.append(loss.item())
      preds = torch.argmax(logits,dim=1).flatten()
      accuracy = (preds == labels).cpu().numpy().mean() * 100
      accuracy1.append(accuracy)
      total_accuracy += accuracy
      loss.backward()
      optimizer.step()
    
    average_train_loss = np.mean(loss1)
    average_train_accuracy = np.mean(accuracy1)
    training_loss.append(average_train_loss)

    if test_dataloader is not None:
      test_loss, test_accuracy = evaluate(model, test_dataloader)
      testing_loss.append(test_loss)
      if test_accuracy>best_accuracy:
        best_accuracy = test_accuracy
        torch.save(model,path)
      time_elapsed = time.time()-t0
      print(f"{i+1:^6} | {average_train_loss:^12.6f} | {average_train_accuracy:^14.6f} | {test_loss:^10.6f} | {test_accuracy:^10.6f} | {time_elapsed:^9.3f}")

  print(f"Training Complete!! \nBest accuracy reported on test set: {best_accuracy:.2f}%\nThis model was saved!!!")
  print("\n")
  return training_loss, testing_loss

def plot_loss_curves(path,train_loss,test_loss,num_of_epochs):
  epch = list(range(num_of_epochs))
  plt.figure(figsize=(15,10))
  plt.plot(epch, train_loss)
  plt.plot(epch, test_loss)
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.legend(['Training Loss', 'Testing Loss'])
  plt.title('Loss vs Epochs')
  # plt.savefig(path)
  plt.show()

def illustrate_attention_weights(model,dataloader,word2idx,path):
  model.eval()
  count = 0
  target_atn, tar, con, context_atn, pred, true_label = None, None, None, None, None, None
  context_words, target_words = list(), list()
  test_accuracy, test_loss = list(), list()
  word_list = list(word2idx.keys())
  # print(word_list[27])
  for batch in dataloader:
    context, target, labels, context_mask, target_mask = tuple(t.to(device) for t in batch)
    logits = model(context,target,context_mask,target_mask)
    preds = torch.argmax(logits,dim=1).flatten()
    true_label = labels[:10].detach().cpu().numpy()
    pred = preds[:10].detach().cpu().numpy()
    target_atn = model.target_atn[:10].detach().cpu().numpy()
    context_atn = model.context_atn[:10].detach().cpu().numpy()
    tar = target[:10].cpu().numpy()
    con = context[:10].cpu().numpy()
    count+=1
    if count==1:
      break

  for i in range(len(tar)):
    words = list()
    for idx in range(len(tar[i])):
      index=tar[i][idx]
      word = word_list[tar[i][idx]]
      if word != "<pad>":
        words.append(word)
    target_words.append(words)

  for i in range(len(con)):
    words = list()
    for idx in range(len(con[i])):
      index=con[i][idx]
      word = word_list[con[i][idx]]
      if word != "<pad>":
        words.append(word)
    context_words.append(words)
  
  final_context_atn, final_target_atn = list(), list()

  for i in range(len(context_atn)):
    length_con = len(context_words[i])
    atn = context_atn[i][0][:length_con]
    final_context_atn.append(atn)

  for i in range(len(target_atn)):
    length_con = len(target_words[i])
    atn = target_atn[i][0][:length_con]
    final_target_atn.append(atn)

  for i in range(10):
    path1 = path+"/target"+str(i)+".jpg"
    path2 = path+"/context"+str(i)+".jpg"
    X1 = final_target_atn[i]
    w1 = target_words[i]
    plt.figure(figsize=(7,1))
    sn.heatmap([X1],cmap='YlGnBu',linewidths=0.5,linecolor="black",xticklabels=w1,yticklabels=False)
    # plt.savefig(path1)
    plt.show()
    X2 = final_context_atn[i][:-1]
    w2 = context_words[i][:-1]
    plt.figure(figsize=(20,1))
    sn.heatmap([X2],cmap='YlGnBu',linewidths=0.5,linecolor="black",xticklabels=w2,yticklabels=False)
    # plt.savefig(path2)
    plt.show()

  # print('\n')
  # print(f"{'Context':^50} | {'Target':^10} | {'Output':^8} | {'True Label':^8}")
  # print("-"*120)
  # polarity = {0:"negative", 1:"neutral", 2:"positive"}
  # for i in range(len(pred)):
  #   complete_context = ' '.join(context_words[i][:-1])
  #   complete_target = ' '.join(target_words[i])
  #   output = polarity[pred[i]]
  #   true_l = polarity[true_label[i]]
  #   print(f"{complete_context:^50} | {complete_target:^10} | {output:^8} | {true_l:^8}\n")

def display_prediction(model,dataloader,word2idx):
  model.eval()
  count = 0
  target_atn, tar, con, context_atn, pred, true_label = None, None, None, None, None, None
  context_words, target_words = list(), list()
  test_accuracy, test_loss = list(), list()
  word_list = list(word2idx.keys())
  # print(word_list[27])
  for batch in dataloader:
    context, target, labels, context_mask, target_mask = tuple(t.to(device) for t in batch)
    logits = model(context,target,context_mask,target_mask)
    preds = torch.argmax(logits,dim=1).flatten()
    true_label = labels[:10].detach().cpu().numpy()
    pred = preds[:10].detach().cpu().numpy()
    tar = target[:10].cpu().numpy()
    con = context[:10].cpu().numpy()
    count+=1
    if count==1:
      break
  for i in range(len(tar)):
    words = list()
    for idx in range(len(tar[i])):
      index=tar[i][idx]
      word = word_list[tar[i][idx]]
      if word != "<pad>":
        words.append(word)
    target_words.append(words)

  for i in range(len(con)):
    words = list()
    for idx in range(len(con[i])):
      index=con[i][idx]
      word = word_list[con[i][idx]]
      if word != "<pad>":
        words.append(word)
    context_words.append(words)
  
  print('\n')
  print(f"{'Context':^50} | {'Target':^10} | {'Output':^8} | {'True Label':^8}")
  print("-"*120)
  polarity = {0:"negative", 1:"neutral", 2:"positive"}
  for i in range(len(pred)):
    complete_context = ' '.join(context_words[i][:-1])
    complete_target = ' '.join(target_words[i])
    output = polarity[pred[i]]
    true_l = polarity[true_label[i]]
    print(f"{complete_context:^50} | {complete_target:^10} | {output:^8} | {true_l:^8}\n")

"""## Restaurant Dataset"""

r_train_context, r_train_target, r_train_polarity = np.array(restaurant_train['context']), np.array(restaurant_train['target']), np.array(restaurant_train['polarity'])
r_test_context, r_test_target, r_test_polarity = np.array(restaurant_test['context']), np.array(restaurant_test['target']), np.array(restaurant_test['polarity'])

rtrain_tokenized_context, rtrain_tokenized_target,rtrain_context_lens, rtrain_target_lens, rtest_tokenized_context, rtest_tokenized_target,rtest_context_lens, rtest_target_lens, word2idx, max_len_context, max_len_target = tokenize_and_build_vocabulary(r_train_context, r_train_target, r_test_context, r_test_target)
rtrain_encoded_context = encode_tokenized_text(rtrain_tokenized_context, word2idx, max_len_context)
rtest_encoded_context = encode_tokenized_text(rtest_tokenized_context, word2idx, max_len_context)
rtrain_encoded_target = encode_tokenized_text(rtrain_tokenized_target, word2idx, max_len_target)
rtest_encoded_target = encode_tokenized_text(rtest_tokenized_target, word2idx, max_len_target)
r_embedding_matrix = load_pretrained_vectors(word2idx,embedding,300)
r_embedding_matrix = torch.tensor(r_embedding_matrix)

r_train_dataset = ABSADataset(rtrain_encoded_context, rtrain_encoded_target, r_train_polarity,rtrain_context_lens, rtrain_target_lens,max_len_context, max_len_target)
r_train_loader = DataLoader(dataset=r_train_dataset, batch_size=128, shuffle=True, num_workers=2)

r_test_dataset = ABSADataset(rtest_encoded_context, rtest_encoded_target, r_test_polarity,rtest_context_lens, rtest_target_lens,max_len_context, max_len_target)
r_test_loader = DataLoader(dataset=r_test_dataset, batch_size=128, shuffle=True, num_workers=2)

loss_fn = nn.CrossEntropyLoss()

def set_seed(seed_value=42):
  random.seed(seed_value)
  np.random.seed(seed_value)
  torch.manual_seed(seed_value)
  torch.cuda.manual_seed_all(seed_value)

set_seed(seed_value=42)
ian_model_with_attn, optimizer = initialize_model(pretrained_embedding = r_embedding_matrix,freeze_embedding = False,hidden_dim=300,
                             num_classes=3,dropout=0.5,max_len_context=max_len_context, max_len_target=max_len_target, l2_reg=0,
                             attention_flag=True,learning_rate=0.001)
epoch = 100
training_loss, testing_loss = train(model=ian_model_with_attn, optimizer=optimizer,train_dataloader=r_train_loader,test_dataloader=r_test_loader, epochs=epoch,path='/content/drive/MyDrive/DL/rian_model_with_attn.pth')
ian_model_with_attn = torch.load('/content/drive/MyDrive/DL/rian_model_with_attn.pth')
test_loss, test_accuracy = evaluate(model=ian_model_with_attn,test_dataloader=r_test_loader)
train_loss, train_accuracy = evaluate(model=ian_model_with_attn,test_dataloader=r_train_loader)
plot_loss_curves(path = "/content/drive/MyDrive/DL/rloss-epoch1.jpg", train_loss=training_loss, test_loss=testing_loss, num_of_epochs=epoch)
print('\n')

print(f"Train Loss: {train_loss:.6f}")
print(f"Training Accuracy: {train_accuracy:.3f}%")
print("\n\n")
print(f"Test Loss: {test_loss:.6f}")
print(f"Testing Accuracy: {test_accuracy:.3f}%")
print('\n')

print("Five Samples from Train Dataset")
illustrate_attention_weights(model=ian_model_with_attn, dataloader=r_train_loader, word2idx=word2idx,path="/content/drive/MyDrive/DL/Train/Restaurant")
print("Five Sample from Test Dataset")
illustrate_attention_weights(model=ian_model_with_attn, dataloader=r_test_loader, word2idx=word2idx,path="/content/drive/MyDrive/DL/Test/Restaurant")

set_seed(seed_value=42)
ian_model_without_attn, optimizer = initialize_model(pretrained_embedding = r_embedding_matrix,freeze_embedding = False,hidden_dim=300,
                             num_classes=3,dropout=0.5,max_len_context=max_len_context, max_len_target=max_len_target, l2_reg=0,
                             attention_flag=False,learning_rate=0.001)
epoch = 100
training_loss, testing_loss = train(model=ian_model_without_attn, optimizer=optimizer,train_dataloader=r_train_loader,test_dataloader=r_test_loader, epochs=epoch,path='/content/drive/MyDrive/DL/rian_model_without_attn.pth')
ian_model_without_attn = torch.load('/content/drive/MyDrive/DL/rian_model_without_attn.pth')
test_loss, test_accuracy = evaluate(model=ian_model_without_attn,test_dataloader=r_test_loader)
train_loss, train_accuracy = evaluate(model=ian_model_without_attn,test_dataloader=r_train_loader)
plot_loss_curves(path = "/content/drive/MyDrive/DL/rloss-epoch2.jpg", train_loss=training_loss, test_loss=testing_loss, num_of_epochs=epoch)

print(f"Train Loss: {train_loss:.6f}")
print(f"Training Accuracy: {train_accuracy:.3f}%")
print("\n\n")
print(f"Test Loss: {test_loss:.6f}")
print(f"Testing Accuracy: {test_accuracy:.3f}%")
print('\n')

set_seed(seed_value=42)
ian_model_with_attn, optimizer = initialize_model(pretrained_embedding = r_embedding_matrix,freeze_embedding = False,hidden_dim=300,
                             num_classes=3,dropout=0.5,max_len_context=max_len_context, max_len_target=max_len_target, l2_reg=0.00001,
                             attention_flag=True,learning_rate=0.001)
epoch = 100
training_loss, testing_loss = train(model=ian_model_with_attn, optimizer=optimizer,train_dataloader=r_train_loader,test_dataloader=r_test_loader, epochs=epoch,path='/content/drive/MyDrive/DL/rian_model_with_attn_l2.pth')
ian_model_with_attn = torch.load('/content/drive/MyDrive/DL/rian_model_with_attn_l2.pth')
test_loss, test_accuracy = evaluate(model=ian_model_with_attn,test_dataloader=r_test_loader)
train_loss, train_accuracy = evaluate(model=ian_model_with_attn,test_dataloader=r_train_loader)
plot_loss_curves(path = "/content/drive/MyDrive/DL/rloss-epoch1.jpg", train_loss=training_loss, test_loss=testing_loss, num_of_epochs=epoch)
print('\n')

print(f"Train Loss: {train_loss:.6f}")
print(f"Training Accuracy: {train_accuracy:.3f}%")
print("\n\n")
print(f"Test Loss: {test_loss:.6f}")
print(f"Testing Accuracy: {test_accuracy:.3f}%")
print('\n')

print("Five Samples from Train Dataset")
illustrate_attention_weights(model=ian_model_with_attn, dataloader=r_train_loader, word2idx=word2idx,path="/content/drive/MyDrive/DL/Train/Restaurant/L2")
print("Five Sample from Test Dataset")
illustrate_attention_weights(model=ian_model_with_attn, dataloader=r_test_loader, word2idx=word2idx,path="/content/drive/MyDrive/DL/Test/Restaurant/L2")

set_seed(seed_value=42)
ian_model_without_attn, optimizer = initialize_model(pretrained_embedding = r_embedding_matrix,freeze_embedding = False,hidden_dim=300,
                             num_classes=3,dropout=0.5,max_len_context=max_len_context, max_len_target=max_len_target, l2_reg=0.00001,
                             attention_flag=False,learning_rate=0.001)
epoch = 100
training_loss, testing_loss = train(model=ian_model_without_attn, optimizer=optimizer,train_dataloader=r_train_loader,test_dataloader=r_test_loader, epochs=epoch,path='/content/drive/MyDrive/DL/rian_model_without_attn_l2.pth')
ian_model_without_attn = torch.load('/content/drive/MyDrive/DL/rian_model_without_attn_l2.pth')
test_loss, test_accuracy = evaluate(model=ian_model_without_attn,test_dataloader=r_test_loader)
test_loss, test_accuracy = evaluate(model=ian_model_without_attn,test_dataloader=r_train_loader)
plot_loss_curves(path = "/content/drive/MyDrive/DL/rloss-epoch2.jpg", train_loss=training_loss, test_loss=testing_loss, num_of_epochs=epoch)

print(f"Train Loss: {train_loss:.6f}")
print(f"Training Accuracy: {train_accuracy:.3f}%")
print("\n\n")
print(f"Test Loss: {test_loss:.6f}")
print(f"Testing Accuracy: {test_accuracy:.3f}%")
print('\n')

"""## Laptop Dataset"""

l_train_context, l_train_target, l_train_polarity = np.array(laptop_train['context']), np.array(laptop_train['target']), np.array(laptop_train['polarity'])
l_test_context, l_test_target, l_test_polarity = np.array(laptop_test['context']), np.array(laptop_test['target']), np.array(laptop_test['polarity'])

ltrain_tokenized_context, ltrain_tokenized_target,ltrain_context_lens, ltrain_target_lens, ltest_tokenized_context, ltest_tokenized_target,ltest_context_lens, ltest_target_lens, lword2idx, lmax_len_context, lmax_len_target = tokenize_and_build_vocabulary(l_train_context, l_train_target, l_test_context, l_test_target)
ltrain_encoded_context = encode_tokenized_text(ltrain_tokenized_context, lword2idx, lmax_len_context)
ltest_encoded_context = encode_tokenized_text(ltest_tokenized_context, lword2idx, lmax_len_context)
ltrain_encoded_target = encode_tokenized_text(ltrain_tokenized_target, lword2idx, lmax_len_target)
ltest_encoded_target = encode_tokenized_text(ltest_tokenized_target, lword2idx, lmax_len_target)
l_embedding_matrix = load_pretrained_vectors(lword2idx,embedding,300)
l_embedding_matrix = torch.tensor(l_embedding_matrix)

l_train_dataset = ABSADataset(ltrain_encoded_context, ltrain_encoded_target, l_train_polarity,ltrain_context_lens, ltrain_target_lens,lmax_len_context, lmax_len_target)
l_train_loader = DataLoader(dataset=l_train_dataset, batch_size=128, shuffle=True, num_workers=2)

l_test_dataset = ABSADataset(ltest_encoded_context, ltest_encoded_target, l_test_polarity,ltest_context_lens, ltest_target_lens,lmax_len_context, lmax_len_target)
l_test_loader = DataLoader(dataset=l_test_dataset, batch_size=128, shuffle=True, num_workers=2)

loss_fn = nn.CrossEntropyLoss()

set_seed(seed_value=42)
lian_model_with_attn, optimizer = initialize_model(pretrained_embedding = l_embedding_matrix,freeze_embedding = False,hidden_dim=300,
                             num_classes=3,dropout=0.5,max_len_context=lmax_len_context, max_len_target=lmax_len_target, l2_reg=0,
                             attention_flag=True,learning_rate=0.001)
epoch = 100
ltraining_loss, ltesting_loss = train(model=lian_model_with_attn, optimizer=optimizer,train_dataloader=l_train_loader,test_dataloader=l_test_loader, epochs=epoch,path="/content/drive/MyDrive/DL/lian_model_with_attn.pth")
lian_model_with_attn = torch.load('/content/drive/MyDrive/DL/lian_model_with_attn.pth')
test_loss, test_accuracy = evaluate(model=lian_model_with_attn,test_dataloader=l_test_loader)
train_loss, train_accuracy = evaluate(model=lian_model_with_attn,test_dataloader=l_train_loader)
plot_loss_curves(path = "/content/drive/MyDrive/DL/lloss-epoch1.jpg", train_loss=ltraining_loss, test_loss=ltesting_loss, num_of_epochs=epoch)
print('\n')

print(f"Train Loss: {train_loss:.6f}")
print(f"Training Accuracy: {train_accuracy:.3f}%")
print("\n\n")
print(f"Test Loss: {test_loss:.6f}")
print(f"Testing Accuracy: {test_accuracy:.3f}%")
print('\n')

print("Five Samples from Train Dataset")
illustrate_attention_weights(model=lian_model_with_attn, dataloader=l_train_loader, word2idx=lword2idx,path="/content/drive/MyDrive/DL/Train/Laptop")
print("Five Sample from Test Dataset")
illustrate_attention_weights(model=lian_model_with_attn, dataloader=l_test_loader, word2idx=lword2idx,path="/content/drive/MyDrive/DL/Test/Laptop")

set_seed(seed_value=42)
lian_model_without_attn, optimizer = initialize_model(pretrained_embedding = l_embedding_matrix,freeze_embedding = False,hidden_dim=300,
                             num_classes=3,dropout=0.5,max_len_context=lmax_len_context, max_len_target=lmax_len_target, l2_reg=0,
                             attention_flag=False,learning_rate=0.001)
epoch = 100
ltraining_loss, ltesting_loss = train(model=lian_model_without_attn, optimizer=optimizer,train_dataloader=l_train_loader,test_dataloader=l_test_loader, epochs=epoch,path="/content/drive/MyDrive/DL/lian_model_without_attn.pth")
lian_model_without_attn = torch.load('/content/drive/MyDrive/DL/lian_model_without_attn.pth')
test_loss, test_accuracy = evaluate(model=lian_model_without_attn,test_dataloader=l_test_loader)
train_loss, train_accuracy = evaluate(model=lian_model_without_attn,test_dataloader=l_train_loader)
plot_loss_curves(path = "/content/drive/MyDrive/DL/lloss-epoch2.jpg", train_loss=ltraining_loss, test_loss=ltesting_loss, num_of_epochs=epoch)
print('\n')

print(f"Train Loss: {train_loss:.6f}")
print(f"Training Accuracy: {train_accuracy:.3f}%")
print("\n\n")
print(f"Test Loss: {test_loss:.6f}")
print(f"Testing Accuracy: {test_accuracy:.3f}%")
print('\n')

set_seed(seed_value=42)
lian_model_with_attn, optimizer = initialize_model(pretrained_embedding = l_embedding_matrix,freeze_embedding = False,hidden_dim=300,
                             num_classes=3,dropout=0.5,max_len_context=lmax_len_context, max_len_target=lmax_len_target, l2_reg=0.00001,
                             attention_flag=True,learning_rate=0.001)
epoch = 100
ltraining_loss, ltesting_loss = train(model=lian_model_with_attn, optimizer=optimizer,train_dataloader=l_train_loader,test_dataloader=l_test_loader, epochs=epoch,path="/content/drive/MyDrive/DL/lian_model_with_attn_l2.pth")
lian_model_with_attn = torch.load('/content/drive/MyDrive/DL/lian_model_with_attn_l2.pth')
test_loss, test_accuracy = evaluate(model=lian_model_with_attn,test_dataloader=l_test_loader)
train_loss, train_accuracy = evaluate(model=lian_model_with_attn,test_dataloader=l_train_loader)
plot_loss_curves(path = "/content/drive/MyDrive/DL/lloss-epoch1.jpg", train_loss=ltraining_loss, test_loss=ltesting_loss, num_of_epochs=epoch)
print('\n')

print(f"Train Loss: {train_loss:.6f}")
print(f"Training Accuracy: {train_accuracy:.3f}%")
print("\n\n")
print(f"Test Loss: {test_loss:.6f}")
print(f"Testing Accuracy: {test_accuracy:.3f}%")
print('\n')

print("Five Samples from Train Dataset")
illustrate_attention_weights(model=lian_model_with_attn, dataloader=l_train_loader, word2idx=lword2idx,path="/content/drive/MyDrive/DL/Train/Laptop/L2")
print("Five Sample from Test Dataset")
illustrate_attention_weights(model=lian_model_with_attn, dataloader=l_test_loader, word2idx=lword2idx,path="/content/drive/MyDrive/DL/Test/Laptop/L2")

set_seed(seed_value=42)
lian_model_without_attn, optimizer = initialize_model(pretrained_embedding = l_embedding_matrix,freeze_embedding = False,hidden_dim=300,
                             num_classes=3,dropout=0.5,max_len_context=lmax_len_context, max_len_target=lmax_len_target, l2_reg=0.00001,
                             attention_flag=False,learning_rate=0.001)
epoch = 100
ltraining_loss, ltesting_loss = train(model=lian_model_without_attn, optimizer=optimizer,train_dataloader=l_train_loader,test_dataloader=l_test_loader, epochs=epoch,path="/content/drive/MyDrive/DL/lian_model_without_attn_l2.pth")
lian_model_without_attn = torch.load('/content/drive/MyDrive/DL/lian_model_without_attn_l2.pth')
test_loss, test_accuracy = evaluate(model=lian_model_without_attn,test_dataloader=l_test_loader)
train_loss, train_accuracy = evaluate(model=lian_model_without_attn,test_dataloader=l_train_loader)
plot_loss_curves(path = "/content/drive/MyDrive/DL/lloss-epoch2.jpg", train_loss=ltraining_loss, test_loss=ltesting_loss, num_of_epochs=epoch)
print('\n')

print(f"Train Loss: {train_loss:.6f}")
print(f"Training Accuracy: {train_accuracy:.3f}%")
print("\n\n")
print(f"Test Loss: {test_loss:.6f}")
print(f"Testing Accuracy: {test_accuracy:.3f}%")
print('\n')

"""## Test"""

model = torch.load('/content/drive/MyDrive/DL/rian_model_with_attn.pth')
illustrate_attention_weights(model=model, dataloader=r_test_loader, word2idx=word2idx,path="attn")
display_prediction(model=model, dataloader=r_test_loader, word2idx=word2idx)

model = torch.load('/content/drive/MyDrive/DL/rian_model_with_attn_l2.pth')
illustrate_attention_weights(model=model, dataloader=r_test_loader, word2idx=word2idx,path="attn")
display_prediction(model=model, dataloader=r_test_loader, word2idx=word2idx)

model = torch.load('/content/drive/MyDrive/DL/rian_model_without_attn.pth')
display_prediction(model=model, dataloader=r_test_loader, word2idx=word2idx)

model = torch.load('/content/drive/MyDrive/DL/rian_model_without_attn_l2.pth')
display_prediction(model=model, dataloader=r_test_loader, word2idx=word2idx)

model = torch.load('/content/drive/MyDrive/DL/lian_model_with_attn.pth')
illustrate_attention_weights(model=model, dataloader=l_test_loader, word2idx=lword2idx,path="attn")
display_prediction(model=model, dataloader=l_test_loader, word2idx=lword2idx)

model = torch.load('/content/drive/MyDrive/DL/lian_model_with_attn_l2.pth')
illustrate_attention_weights(model=model, dataloader=l_test_loader, word2idx=lword2idx,path="attn")
display_prediction(model=model, dataloader=l_test_loader, word2idx=lword2idx)

model = torch.load('/content/drive/MyDrive/DL/lian_model_without_attn.pth')
display_prediction(model=model, dataloader=l_test_loader, word2idx=lword2idx)

model = torch.load('/content/drive/MyDrive/DL/lian_model_without_attn_l2.pth')
display_prediction(model=model, dataloader=l_test_loader, word2idx=lword2idx)
