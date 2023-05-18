#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/prateekjoshi565/Fine-Tuning-BERT/blob/master/Fine_Tuning_BERT_for_Spam_Classification.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Install Transformers Library

# In[ ]:


#get_ipython().system('pip install transformers')


from lime.lime_text import LimeTextExplainer
from grad import GradCam
from text_attention import text_attention_map 
from text_attention1 import text_attention_map as tam
from shap_heatmap import text_attention_map as shm
from lime_heatmap import text_attention_map as lhm
#from text_attention1 import text_attention_map as tam
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
import transformers
from transformers import AutoModel, BertTokenizerFast

import emoji
# specify GPU
device = torch.device("cuda")

import re
# # Load Dataset

# In[3]:
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import nltk
import re
from nltk.corpus import stopwords
import string
#Removal of Punctuations
punctuations=list(string.punctuation)
# specify GPU
device = torch.device("cuda")

nltk.download('stopwords')
stops = set(stopwords.words("english"))
def cleantext(string):
    text = string.lower().split()
    text = " ".join(text)
    text = re.sub(r"http(\S)+",' ',text)    
    text = re.sub(r"www(\S)+",' ',text)
    text = re.sub(r"&",' and ',text)  
    text = text.replace('&amp',' ')
    text = re.sub(r"[^0-9a-zA-Z]+",' ',text)
    text = text.split()
    text = [w for w in text if not w in stops]
    text = " ".join(text)
    return text
    
    
    
    
    
max_seq_len = 128
batch_size=4
from transformers import AutoTokenizer
# import BERT-base pretrained model
tokenizer = AutoTokenizer.from_pretrained("digitalepidemiologylab/covid-twitter-bert-v2")
print(len(tokenizer))
special_tokens_dict = {'additional_special_tokens': ['covid','covid19']}
tokenizer.add_tokens(['covid','covid19'], special_tokens=True)
print(len(tokenizer))
bert= AutoModel.from_pretrained("digitalepidemiologylab/covid-twitter-bert-v2", output_hidden_states=True,output_attentions=True)


import shap
from lime.lime_text import LimeTextExplainer
from transformers import pipeline
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased", do_lower_case=True)




#pipe = pipeline("text-classification", model=model)

import scipy as sp    


def f(x):
    
    tv = torch.tensor([tokenizer.encode(v, padding='max_length', max_length=128, truncation=True) for v in x]).cuda()
    outputs,_ = model(tv)
    outputs=outputs.detach().cpu().numpy()
    scores = (np.exp(outputs).T / np.exp(outputs).sum(-1)).T
    
    val = sp.special.logit(scores[:,1]) # use one vs rest logit units
    return outputs

# build an explainer using a token masker
explainer = shap.Explainer(f, tokenizer)



lexplainer = LimeTextExplainer(class_names = [0, 1])

def predict_probab(STR):
    
    list_all = []
   
    for strings in STR:
        z = tokenizer.encode_plus(strings, add_special_tokens = True, max_length = 128, truncation = True,padding = 'max_length', return_token_type_ids=True, return_attention_mask = True,  return_tensors = 'np')
        inputs = [z['input_ids'], z['attention_mask']]
        
        v,_=model(torch.LongTensor(z['input_ids']).to(device))
        k = []
        k.append(float(torch.argmax(v,axis=-1).reshape(-1,1)))
        k.append(float(1-torch.argmax(v,axis=-1).reshape(-1,1)))
        k = np.array(k).reshape(1,-1)
        list_all.append(k)
    list_all = np.array(list_all).squeeze()
    return np.array(list_all)

    




def input_fun(df,sampler):
    
    df['tweet']=df['tweet'].str.lower()
    df['tweet']=df['tweet'].str.replace('#','')
    #df['tweet'] = df['tweet'].apply(lambda x: ' '.join([i.strip(' '.join(punctuations)) for i in x.split() if i not in punctuations]))
    df['tweet']=df['tweet'].apply(lambda x: ' '.join([emoji.demojize(word) for word in x.split()])) 
    df['tweet']=df['tweet'].apply(lambda x:''.join([i if ord(i) < 128 else ' ' for i in x]))
    df['tweet']=df['tweet'].map(lambda x: cleantext(x))
    df['label']=df['label'].replace('real',1)
    df['label']=df['label'].replace('fake',0)
    text=df['tweet']
    labels=df['label']
    tokens = tokenizer.batch_encode_plus(text.tolist(),max_length = max_seq_len,pad_to_max_length=True,truncation=True, return_token_type_ids=False)
    seq = torch.tensor(tokens['input_ids'])
    print(seq)
    #mask = torch.tensor(tokens['attention_mask'])
    y = torch.tensor(labels.tolist())
    data = TensorDataset(seq,y)
    data_sampler = sampler(data)
    data_dataloader = DataLoader(data, sampler=data_sampler, batch_size=batch_size)
    return data,data_sampler,data_dataloader

df = pd.read_csv("train1.csv")
#df=pd.read_csv('Constraint_English_Train.csv')
train_data,train_sampler,train_dataloader=input_fun(df,RandomSampler)
df_val=pd.read_csv('valid1.csv')
val_data,val_sampler,val_dataloader=input_fun(df_val,SequentialSampler)
df_test=pd.read_csv('valid1.csv')
test_data,test_sampler,test_dataloader=input_fun(df_test,SequentialSampler)

# In[6]:






# freeze all the parameters
for param in bert.parameters():
    param.requires_grad = True


# # Define Model Architecture
from torch.autograd import Variable
# In[15]:


class BERT_Arch(nn.Module):

    def __init__(self, bert):
      
      super(BERT_Arch, self).__init__()

      self.bert = bert 
      
      # dropout layer
      self.dropout = nn.Dropout(0.2)
      
      # relu activation function
      self.relu =  nn.ReLU()

      # dense layer 1
      self.fc1 = nn.Linear(1024,128)
      
      # dense layer 2 (Output layer)
      self.fc2 = nn.Linear(128,2)

      #softmax activation function
      self.softmax = nn.LogSoftmax(dim=-1)

    #define the forward pass
    def forward(self, sent_id):

      mask = []
      for sent in sent_id:
          att_mask = [int(token_id > 0) for token_id in sent]
          mask.append(att_mask)
      mask=torch.Tensor(mask).to(device)    
      outputs = self.bert(sent_id, attention_mask=mask)
      
      pooled_output = outputs.pooler_output
      x = self.fc1(pooled_output)

      x = self.relu(x)

      x = self.dropout(x)

      # output layer
      x = self.fc2(x)
      
      # apply softmax activation
      x = self.softmax(x)
      
      return x,outputs


# In[16]:

bert.resize_token_embeddings(len(tokenizer))
# pass the pre-trained BERT to our define architecture
model = BERT_Arch(bert)
print(model)
# push the model to GPU
model = model.to(device)
for param in model.parameters():
	param.requires_grad=True

# In[17]:

# optimizer from hugging face transformers
from transformers import AdamW

# define the optimizer
optimizer = AdamW(model.parameters(), lr = 1e-5)



# loss function
cross_entropy  = nn.NLLLoss() 

# number of training epochs
epochs = 10


# # Fine-Tune BERT

# In[21]:

def rescale(input_list):
	the_array = np.asarray(input_list)
	the_max = np.max(the_array)
	the_min = np.min(the_array)
	rescale = (the_array - the_min)/(the_max-the_min)*100
	return rescale.tolist()
	
def attn_grad_fun(a,f,g):
    for p in model.parameters():
        print(p.grad)
        i=i+1
        print(i)
    # update parameters
    optimizer.step()
    g=outputs.hidden_states[1]#model.bert.embeddings.word_embeddings(sent_id)
    i=0
    for p in model.parameters():
        print(p.grad)
        i=i+1
        print(i)
    l=loss.grad
    print(l)
    for i in range(24):
        g=outputs.hidden_states[i]
        for j in range(8):
            attn=outputs.attentions[i][:,j,:,:]
            #l=model.bert.encoder.layer[i].grad
            attn=torch.mean(attn,axis=-1)
            all_att_grad=g*l
            all_att_grad=torch.mean(all_att_grad,axis=-1)
            all_att_grad=attn+all_att_grad
            all_att_grad=rescale(all_att_grad.detach().cpu().numpy())
            #print(all_att_grad)	

# function to train the model
def train(ep):
  
  model.train()

  total_loss, total_accuracy = 0, 0
  
  # empty lit to save model predictions
  total_preds=[]
  attention_gradient=[]
  # iterate over batches
  for step,batch in enumerate(train_dataloader):
    
    # progress update after every 50 batches.
    if step % 50 == 0 and not step == 0:
      print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(train_dataloader)))

    # push the batch to gpu
    batch = [r.to(device) for r in batch]
 
    sent_id,  labels = batch

    # clear previously calculated gradients 
    model.zero_grad()        

    # get model predictions for the current batch
    #rint(sent_id,mask)
    preds,outputs = model(sent_id)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    #print("Hidden States:",outputs.hidden_states)
    # compute the loss between actual and predicted values
    loss = cross_entropy(preds, labels)
    #print(outputs)
    grad=model.bert.embeddings.word_embeddings(sent_id)
    print(outputs.attentions[0].shape)
    for i in range(16):
        outputs.attentions[i].retain_grad()
    # add on to the total loss
    total_loss = total_loss + loss.item()
    #loss.requires_grad=True
    # backward pass to calculate the gradients
    #m=loss.register_hook(lambda grad: grad)
   
    
    loss.backward()
    optimizer.step()
    if ep>=8:
        sentence=tokenizer.batch_decode(sent_id[0])
        asentence=[' '.join(i) for i in [sentence]][0]
        
        asentence=asentence.replace('[CLS]','')
        asentence=asentence.replace('[SEP]','')
        asentence=asentence.replace('[PAD]','')
        
        asentence=" ".join(asentence.split())
        print(sentence)
        print(asentence)
        ls=labels[0]
        lh=[]
        lh_grad=[]
        
        exp = lexplainer.explain_instance(asentence, predict_probab, num_features=128)
        ldf=pd.DataFrame(exp.as_list())
        lhm(ldf.iloc[:,0],rescale(ldf.iloc[:,1]),"results/"+str(step)+"_"+str(epoch)+"_0")
        
        shap_values = explainer(sentence,fixed_context=1)
        q=[]
        for qi in range(len(shap_values.data)):
            q.append(shap_values.data[qi][1])
        shap_d=q  
        q=[]
        for qi in range(len(shap_values.values)):
            q.append(shap_values.values[qi][1][0])
        shap_v=q    
        print(shap_values.values)
        shm(shap_d,rescale(shap_v),"results/"+str(step)+"_"+str(epoch)+"_0")
        q=[]
        for qi in range(len(shap_values.values)):
            q.append(shap_values.values[qi][1][1])
        shap_v=q    
        print(shap_values.values)
        shm(shap_d,rescale(shap_v),"results/"+str(step)+"_"+str(epoch)+"_1")
        for i in range(16):
            h_grad=[]
            h=[]
            for j in range(16):
                g=outputs.attentions[i][0][j]
                #print(g.shape)
                l=outputs.attentions[i].grad[0][j]
                #print(l.shape)
                all_att_grad=g*l
                #print(all_att_grad.shape)
                #print(g.shape)
                all_att_grad=torch.mean(all_att_grad,axis=1)
                #all_att_grad=rescale(all_att_grad.detach().cpu().numpy())
                g=torch.mean(g,axis=1)
                
                h_grad.append(all_att_grad.detach().cpu().numpy())
                h.append(g.detach().cpu().numpy())
                #tam(sentence,rescale(all_att_grad.detach().cpu().numpy()),"results/"+str(step)+"_"+str(epoch)+"_"+str(i)+"_"+str(j))
                #text_attention_map(sentence,rescale(g.detach().cpu().numpy()),"results/"+str(step)+"_"+str(epoch)+"_"+str(i)+"_"+str(j))
            h=np.array(h)
            h_grad=np.array(h_grad)
            #print(h_grad.shape)
            #print(h.shape)
            h=np.mean(h,axis=0) 
            h_grad=np.mean(h_grad,axis=0)  
            lh.append(h)
            lh_grad.append(h_grad)
        lh=np.array(lh)
        lh_grad=np.array(lh_grad)
        #print(lh_grad.shape)
        #print(lh.shape)
        lh=np.mean(lh,axis=0) 
        lh_grad=np.mean(lh_grad,axis=0)        
        tam(sentence,rescale(lh_grad),"results/"+str(step)+"_"+str(epoch))
        text_attention_map(sentence,rescale(lh),"results/"+str(step)+"_"+str(epoch))        
		# model predictions are stored on GPU. So, push it to CPU
    #optimizer.step()
    preds=preds.detach().cpu().numpy()

    # append the model predictions
    total_preds.append(preds)

  # compute the training loss of the epoch
  avg_loss = total_loss / len(train_dataloader)
  
  # predictions are in the form of (no. of batches, size of batch, no. of classes).
  # reshape the predictions in form of (number of samples, no. of classes)
  total_preds  = np.concatenate(total_preds, axis=0)

  #returns the loss and predictions
  return avg_loss, total_preds


# In[22]:


# function for evaluating the model
def evaluate():
  
  print("\nEvaluating...")
  
  # deactivate dropout layers
  model.eval()

  total_loss, total_accuracy = 0, 0
  
  # empty list to save the model predictions
  total_preds = []

  # iterate over batches
  for step,batch in enumerate(val_dataloader):
    
    # Progress update every 50 batches.
    if step % 50 == 0 and not step == 0:
      
      # Calculate elapsed time in minutes.
      #elapsed = format_time(time.time() - t0)
            
      # Report progress.
      print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(val_dataloader)))

    # push the batch to gpu
    batch = [t.to(device) for t in batch]

    sent_id,labels = batch

    # deactivate autograd
    with torch.no_grad():
      
      # model predictions
      preds,outputs = model(sent_id)
      
      # compute the validation loss between actual and predicted values
      loss = cross_entropy(preds,labels)

      total_loss = total_loss + loss.item()

      preds = preds.detach().cpu().numpy()

      total_preds.append(preds)

  # compute the validation loss of the epoch
  avg_loss = total_loss / len(val_dataloader) 

  # reshape the predictions in form of (number of samples, no. of classes)
  total_preds  = np.concatenate(total_preds, axis=0)

  return avg_loss, total_preds


# # Start Model Training

# In[23]:


# set initial loss to infinite
best_valid_loss = float('inf')

# empty lists to store training and validation loss of each epoch
train_losses=[]
valid_losses=[]
epochs=10

for epoch in range(epochs):
     
    print('\n Epoch {:} / {:}'.format(epoch + 1, epochs))
    
    #train model
    train_loss, _ = train(epoch)
    
    #evaluate model
    valid_loss, _ = evaluate()
    
    #save the best model
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'saved_weights_ctbert_attention.pt')
    
    # append training and validation loss
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    
    print(f'\nTraining Loss: {train_loss:.3f}')
    print(f'Validation Loss: {valid_loss:.3f}')


# # Load Saved Model

# In[24]:

#load weights of best model
path = 'saved_weights_ctbert_attention.pt'
model.load_state_dict(torch.load(path))


# # Get Predictions for Test Data

# In[25]:
total_preds=[]
test_y=[]
#from text_attention import text_attention_map
model.eval()

total_loss, total_accuracy = 0, 0
  
# empty list to save the model predictions
total_preds = []

# iterate over batches
for step,batch in enumerate(test_dataloader):
    
    # Progress update every 50 batches.
    if step % 50 == 0 and not step == 0:
      
      # Calculate elapsed time in minutes.
      #elapsed = format_time(time.time() - t0)
            
      # Report progress.
      print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(val_dataloader)))

    # push the batch to gpu
    batch = [t.to(device) for t in batch]

    sent_id, labels = batch

    with torch.no_grad():
	     preds,outputs = model(sent_id)
	     print(outputs)
	     att=outputs.attentions
	     
	     preds = preds.detach().cpu().numpy()
	     total_preds.append(preds)
	     test_y.append(labels.cpu().numpy())
	     text=tokenizer.convert_ids_to_tokens(sent_id[0])
"""  
  print(text)
    
    
  print(len(text)) 
  #att=att[2]
  #print(att.shape)
  from bertvizf.bertviz import head_view
  head_view(att, text)
  att=att[2]
  att=torch.mean(att,axis=1)
  print(att.shape) 
  att=torch.sum(att,axis=-1)
  print(att.shape) 
  text_att=att[0]
  print(text_att.shape)
  text_attention_map(text,text_att)
  """

      
total_preds=np.concatenate(total_preds,axis=0)  
test_y=np.concatenate(test_y,axis=0) 

"""     
# get predictions for test data
with torch.no_grad():
  
  preds = model(test_seq.to(device), test_mask.to(device))
  preds = preds.detach().cpu().numpy()
"""

# In[26]:


# model's performance
preds = np.argmax(total_preds,axis=1)
print(classification_report(test_y, preds,digits=4))
print(accuracy_score(test_y, preds))
print(confusion_matrix(test_y, preds))

# In[27]:


# confusion matrix
pd.crosstab(test_y, preds)


# In[ ]:




