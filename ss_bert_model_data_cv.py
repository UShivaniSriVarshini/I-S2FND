import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
import transformers
from transformers import AutoModel, BertTokenizerFast
import emoji
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler,Sampler, Dataset,BatchSampler
from torch.utils.data import Sampler, Dataset,BatchSampler
from sklearn.model_selection import ShuffleSplit
# specify GPU
import itertools
device = torch.device("cuda")

def create_split_loaders(dataset, split, aug_count, batch_size):
    train_folds_idx = split[0]
    valid_folds_idx = split[1]
    train_sampler = RandomSampler(train_folds_idx)
    valid_sampler = SequentialSampler(valid_folds_idx)
    train_batch_sampler = BatchSampler(train_sampler,batch_size=batch_size,drop_last=True
                                            )
    valid_batch_sampler = BatchSampler(valid_sampler,batch_size=batch_size,drop_last=False
                                      )
    train_loader = DataLoader(dataset, sampler=train_sampler, batch_size=batch_size)
    valid_loader = DataLoader(dataset, sampler=valid_sampler, batch_size=batch_size)
    return (train_loader, valid_loader)    


def get_all_split_loaders(dataset, cv_splits, aug_count=5, batch_size=30):
    """Create DataLoaders for each split.

    Keyword arguments:
    dataset -- Dataset to sample from.
    cv_splits -- Array containing indices of samples to 
                 be used in each fold for each split.
    aug_count -- Number of variations for each sample in dataset.
    batch_size -- batch size.
    
    """
    split_samplers = []
    
    for i in range(len(cv_splits)):
        split_samplers.append(
            create_split_loaders(dataset,
                                 cv_splits[i], 
                                 aug_count, 
                                 batch_size)
        )
    return split_samplers


def data_clean(df):
    df['tweet']=df['tweet'].str.lower()
    df['tweet']=df['tweet'].apply(lambda x: ' '.join([emoji.demojize(word) for word in str(x).split()]))
    df['tweet']=df['tweet'].apply(lambda x:''.join([i if ord(i) < 128 else ' ' for i in x]))
    df['label']=df['label'].replace('real',1)
    df['label']=df['label'].replace('fake',0)
    df=df[['tweet','label']]
    return df


df = pd.read_csv('./covid_dataset/complete_data/processed_data/train/train.txt',header=None , sep='\t')
df['tweet']=df.iloc[:,1]
df['label']=df.iloc[:,0]
df=df[['tweet','label']]

df_labelled=data_clean(df)


df_unlabelled=pd.read_csv('./covid_dataset/complete_data/processed_data/unlabelled.txt',header=None , sep='\t')
df_val_1=pd.DataFrame()
df_val_1['tweet']=df_unlabelled.iloc[:,1]
df_val_1['label']=df_unlabelled.iloc[:,0]
df_unlabelled=df_val_1

df_unlabelled=data_clean(df_unlabelled)

df_test=pd.read_csv('./covid_dataset/complete_data/processed_data/test/test.txt',header=None , sep='\t')
df_test['tweet']=df_test.iloc[:,1]
df_test['label']=df_test.iloc[:,0]
df_test=data_clean(df_test)



train_text, validating_text, train_labels, validating_labels = train_test_split(df_labelled['tweet'], df_labelled['label'], 
                                                                    random_state=2018, 
                                                                    test_size=0.2, 
                                                                    stratify=df['label'])

print(len(train_text))                                                                    
splitter = ShuffleSplit(n_splits=5, test_size=.9, random_state=0)

splits = []
for train_idx, test_idx in splitter.split(train_text, train_labels):
    splits.append((train_idx, test_idx))                                                                    

print(len(splits))


train_text_L1, temp_text, train_labels_L1, temp_labels = train_test_split(train_text,  train_labels, 
                                                                    random_state=2018, 
                                                                    test_size=0.5, 
                                                                    stratify=train_labels)

train_text_L2, train_text_L3, train_labels_L2, train_labels_L3 = train_test_split(temp_text, temp_labels, 
                                                                random_state=2018, 
                                                                test_size=0.5, 
                                                                stratify=temp_labels)
                                                                



unlabelled_text=df_unlabelled['tweet']
test_text,test_labels=df_test['tweet'],df_test['label']

#Loading pretrained model and tokenizer
bert = AutoModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

max_seq_len = 128


def data_loader_creation(t,l,sampler):
    tokens_v = tokenizer.batch_encode_plus(
    t.tolist(),
    max_length = max_seq_len,
    pad_to_max_length=True,
    truncation=True,
    return_token_type_ids=False)
    v_seq = torch.tensor(tokens_v['input_ids'])
    v_mask = torch.tensor(tokens_v['attention_mask'])
    #v_y = torch.tensor(l.tolist())
    
    #define a batch size
    batch_size = 32
    if len(l)>1:
        v_y = torch.tensor(l.tolist())
        val_data = TensorDataset(v_seq, v_mask,v_y)
    else:
        val_data = TensorDataset(v_seq, v_mask)
    val_sampler = sampler(val_data)
    
    
    dataloaders = get_all_split_loaders(val_data, splits, aug_count=5, batch_size=10)

    
    v_dataloader = DataLoader(val_data, sampler = val_sampler, batch_size=batch_size)
    print(v_dataloader)
    return val_data

val_data=data_loader_creation(train_text,train_labels,Sampler)
dataloaders = get_all_split_loaders(val_data, splits, aug_count=5, batch_size=10)
#train_dataloader_L1=data_loader_creation(train_text_L1,train_labels_L1,RandomSampler)
#train_dataloader_L2=data_loader_creation(train_text_L2,train_labels_L2,RandomSampler)
#train_dataloader_L3=data_loader_creation(train_text_L3,train_labels_L3,RandomSampler)
#train_dataloader_U=data_loader_creation(unlabelled_text,'0',RandomSampler)
valid_dataloader=data_loader_creation(validating_text,validating_labels,SequentialSampler)
test_dataloader=data_loader_creation(test_text,test_labels,SequentialSampler)


for param in bert.parameters():
    param.requires_grad = True



class BERT_Arch(nn.Module):

    def __init__(self, bert):
      
      super(BERT_Arch, self).__init__()

      self.bert = bert 
     

    #define the forward pass
    def forward(self, sent_id, mask):

      #pass the inputs to the model  
      cls_hs = self.bert(sent_id, attention_mask=mask)
      
      x = cls_hs[0]

      
      return x



class _head(nn.Module):

    def __init__(self,hidden_size,n_layers,bd,do):
      
      super(_head, self).__init__()

      
      
      # dropout layer
      self.dropout = nn.Dropout(0.2)
      self.lstm = nn.LSTM(768, 
                           hidden_size, 
                           num_layers=n_layers, 
                           bidirectional=bd, 
                           dropout=do,
                           batch_first=True)
      # relu activation function
      self.relu =  nn.ReLU()

      # dense layer 1
      self.fc1 = nn.Linear(768,256)
      self.fc4 = nn.Linear(hidden_size*2,2)
      #softmax activation function
      self.softmax = nn.LogSoftmax()

    #define the forward pass
    def forward(self, x):

      
      packed_output, (hidden, cell) = self.lstm(x) 
      x = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)  
      

      # output layer
      x = self.fc4(x)
      # apply softmax activation
      x = self.softmax(x)

      return x

# In[16]:


# pass the pre-trained BERT to our define architecture
model = BERT_Arch(bert)
head1=_head(256,1,True,0.2)
head2=_head(128,2,True,0.2)
head3=_head(256,2,False,0.2)
# push the model to GPU
model = model.to(device)
head1=head1.to(device)
head2=head2.to(device)
head3=head3.to(device)

model.load_state_dict(torch.load('model.pt'))
head1.load_state_dict(torch.load('head1.pt'))
head2.load_state_dict(torch.load('head2.pt'))
head3.load_state_dict(torch.load('head3.pt'))

# optimizer from hugging face transformers
from transformers import AdamW

# define the optimizer
optimizer1 = AdamW(list(model.parameters())+list(head1.parameters()), lr = 1e-5)
optimizer2 = AdamW(list(model.parameters())+list(head2.parameters()), lr = 1e-5)
optimizer3 = AdamW(list(model.parameters())+list(head3.parameters()), lr = 1e-5)

# loss function
cross_entropy  = nn.NLLLoss() 



valid_acc=float(0)
# function for evaluating the model
def validating_data_model(v_dataloader):
  
  print("\nEvaluating...")
  
  model.eval()

  total_loss, total_accuracy = 0, 0
  samples=[]
  l=[]
  predictions=[]
  vacc=0
  for step,batch in enumerate(v_dataloader):
    
        if step % 50 == 0 and not step == 0:
      
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(v_dataloader)))

        # push the batch to gpu
        batch = [t.to(device) for t in batch]

        sent_id, mask,labels= batch
        
        with torch.no_grad():
            preds = model(sent_id, mask)
            h1=head1(preds)
            h2=head2(preds)
            h3=head3(preds)
            h1=np.argmax(h1.detach().cpu().numpy(),axis=1)
            h2=np.argmax(h2.detach().cpu().numpy(),axis=1)
            h3=np.argmax(h3.detach().cpu().numpy(),axis=1)
            all_outputs=np.array([h1,h2,h3]).T
            from scipy.stats import mode
            predictions.append(mode(all_outputs, axis=1)[0])
            samples.append(predictions)
            l.append(labels.cpu().numpy())
            
  predictions=np.concatenate(predictions,axis=0)
  l=np.concatenate(l,axis=0)
  return predictions,l
  
  


# function to train the model
def train(train_batch_dataloader,optimizer,head):
  
  model.train()
  valid_acc=float(0)
  total_loss, total_accuracy = 0, 0
  
  # empty list to save model predictions
  total_preds=[]
  act_train=[]
  print(train_batch_dataloader)
  # iterate over batches
  for batch in train_batch_dataloader:
    
    print("training entered")

    # push the batch to gpu
    batch = [r.to(device) for r in batch]
 
    sent_id, mask, labels = batch
    labels=labels.long()
    # clear previously calculated gradients 
    model.zero_grad()        

    # get model predictions for the current batch
    #rint(sent_id,mask)
    preds = model(sent_id, mask)
    preds=head(preds)
    
    # compute the loss between actual and predicted values
    loss = cross_entropy(preds, labels)

    # add on to the total loss
    total_loss = total_loss + loss.item()

    # backward pass to calculate the gradients
    loss.backward()

    # clip the the gradients to 1.0. It helps in preventing the exploding gradient problem
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    # update parameters
    optimizer.step()

    # model predictions are stored on GPU. So, push it to CPU
    preds=preds.detach().cpu().numpy()

    # append the model predictions
    total_preds.append(preds)
    act_train.append(labels.cpu().numpy())
  # compute the training loss of the epoch
  avg_loss = total_loss / len(train_dataloader)
  
  # predictions are in the form of (no. of batches, size of batch, no. of classes).
  # reshape the predictions in form of (number of samples, no. of classes)
  total_preds  = np.concatenate(total_preds, axis=0)
  act_train  = np.concatenate(act_train, axis=0)
  
  #validating_data_model(v_dataloader)
  return avg_loss, total_preds,act_train


# In[22]:


# function for evaluating the model
def evaluate(val_dataloader,t1,t2,t3):
  
  print("\nEvaluating...")
  
  model.eval()

  total_loss, total_accuracy = 0, 0
  samples=[]
  masks=[]
  predictions=[]
  l=[]
  for step,batch in enumerate(val_dataloader):
    
        if step % 50 == 0 and not step == 0:
      
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(val_dataloader)))

        # push the batch to gpu
        batch = [t.to(device) for t in batch]

        sent_id, mask= batch
        
        with torch.no_grad():
            preds = model(sent_id, mask)
            h1=head1(preds)
            h2=head2(preds)
            h3=head3(preds)
            h1=np.argmax(h1.detach().cpu().numpy(),axis=1)
            h2=np.argmax(h2.detach().cpu().numpy(),axis=1)
            h3=np.argmax(h3.detach().cpu().numpy(),axis=1)
            all_outputs=np.array([h1,h2,h3]).T
            from scipy.stats import mode
            predictions.append(mode(all_outputs, axis=1)[0])
            samples.append(sent_id.cpu().numpy())
            masks.append(mask.cpu().numpy())
  for step,batch in enumerate(t1):
    
        batch = [t.to(device) for t in batch]

        sent_id, mask,labels= batch
        samples.append(sent_id.cpu().numpy())
        masks.append(mask.cpu().numpy()) 
        l.append(labels.cpu().numpy())  
  for step,batch in enumerate(t2):
    
        batch = [t.to(device) for t in batch]

        sent_id, mask,labels= batch
        samples.append(sent_id.cpu().numpy())
        masks.append(mask.cpu().numpy()) 
        l.append(labels.cpu().numpy()) 
  for step,batch in enumerate(t3):
    
        batch = [t.to(device) for t in batch]

        sent_id, mask,labels= batch
        samples.append(sent_id.cpu().numpy())
        masks.append(mask.cpu().numpy()) 
        l.append(labels.cpu().numpy())    
  predictions =  torch.Tensor(np.concatenate(predictions, axis=0))
  samples = torch.Tensor( np.concatenate(samples, axis=0) )       
  masks =  torch.Tensor(np.concatenate(masks, axis=0) )
  l=  torch.Tensor(np.concatenate(l, axis=0) )
  l=l.reshape(l.shape[0],1)
  print(predictions.shape,l.shape)
  predictions= torch.Tensor(np.concatenate([predictions,l], axis=0) )
  # wrap tensors
  val_data = TensorDataset(samples, masks, predictions)

  # sampler for sampling the data during training
  val_sampler = SequentialSampler(val_data)

  # dataLoader for validation set
  val_dataloader_1 = DataLoader(val_data, sampler = val_sampler, batch_size=32)
  #train_dataloader= val_dataloader_1
  return val_dataloader_1,predictions,samples,masks,predictions


def test(test_dataloader):
  
  print("\nTesting...")
  
  model.eval()

  total_loss, total_accuracy = 0, 0
  samples=[]
  masks=[]
  l=[]
  predictions=[]
  for step,batch in enumerate(test_dataloader):
    
        if step % 50 == 0 and not step == 0:
      
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(val_dataloader)))

        # push the batch to gpu
        batch = [t.to(device) for t in batch]

        sent_id, mask, labels= batch
        
        with torch.no_grad():
            preds = model(sent_id, mask)
            h1=head1(preds)
            h2=head2(preds)
            h3=head3(preds)
            h1=np.argmax(h1.detach().cpu().numpy(),axis=1)
            h2=np.argmax(h2.detach().cpu().numpy(),axis=1)
            h3=np.argmax(h3.detach().cpu().numpy(),axis=1)
            all_outputs=np.array([h1,h2,h3]).T
            from scipy.stats import mode
            predictions.append(mode(all_outputs, axis=1)[0])
            samples.append(sent_id.cpu().numpy())
            masks.append(mask.cpu().numpy())
            l.append(labels.cpu().numpy())
  predictions =  np.concatenate(predictions, axis=0)
  samples =  np.concatenate(samples, axis=0)        
  masks =  np.concatenate(masks, axis=0) 
  l =  np.concatenate(l, axis=0) 
  
  return test_dataloader,predictions,l



# set initial loss to infinite
best_valid_loss = float('inf')

# empty lists to store training and validation loss of each epoch
train_losses=[]
valid_losses=[]
epochs=3
#for each epoch
train_acc=[]
val_acc=[]

import time
start_time = time.time()
batch_size=32
"""
tu=train_dataloader_U
t1=train_dataloader_L1
t2=train_dataloader_L2
t3=train_dataloader_L3"""
for train_dataloader in dataloaders:
	print(train_dataloader)
	print(train_dataloader[0])
	for epoch in range(epochs):
		valid_acc=float(0)
		if epoch<=(epochs/2): 
		    print('\n Epoch {:} / {:}'.format(epoch + 1, epochs))
		    train_loss,p,a= train(train_dataloader[0],optimizer1,head1)
		    
		    p=np.argmax(p,axis=1)
		    train_acc.append(accuracy_score(p,a))
		    
		else:
		    
		    _,predictions,s,m,l=evaluate(tu,t1,t2,t3)	
		    s1, s2,m1,m2,l1,l2= train_test_split(s,m,l,
		                                                                random_state=2018, 
		                                                                test_size=0.66, 
		                                                                stratify=l)
		                                                                
		    s2,s3,m2,m3,l2,l3= train_test_split(s2,m2,l2,
		                                                                random_state=2018, 
		                                                                test_size=0.33, 
		                                                                stratify=l2) 
		                                                                
		                                                                   

		    l1=l1.reshape(l1.shape[0],)
		    l2=l2.reshape(l2.shape[0],)
		    l3=l3.reshape(l3.shape[0],)
		    s1,s2,s3,m1,m2,m3,l1,l2,l3=s1.long(),s2.long(),s3.long(),m1.long(),m2.long(),m3.long(),l1.long(),l2.long(),l3.long()
		    val_data = TensorDataset(s1, m1, l1)
		    val_sampler = RandomSampler(val_data)
		    train_dataloader_L1 = DataLoader(val_data, sampler = val_sampler, batch_size=batch_size)
		    val_data = TensorDataset(s2, m2, l2)

		    val_sampler = RandomSampler(val_data)
		    train_dataloader_L2 = DataLoader(val_data, sampler = val_sampler, batch_size=batch_size)
		    val_data = TensorDataset(s3, m3, l3)

		    val_sampler = RandomSampler(val_data)
		    train_dataloader_L3 = DataLoader(val_data, sampler = val_sampler, batch_size=batch_size)
		    train_loss,p,a= train(train_dataloader_L1,optimizer1,head1)
		    train_loss,p,a= train(train_dataloader_L2,optimizer2,head2)
		    train_loss,p,a= train(train_dataloader_L3,optimizer3,head3)
		    validating_data_model(valid_dataloader)
		    vacc=accuracy_score(l,predictions)
		    if valid_acc<vacc:
		       valid_acc=vacc
		       torch.save(model.state_dict(), 'model.pt')
		       torch.save(head1.state_dict(), 'head1.pt')
		       torch.save(head2.state_dict(), 'head2.pt')
		       torch.save(head3.state_dict(), 'head3.pt')
 
model.load_state_dict(torch.load('model.pt'))
head1.load_state_dict(torch.load('head1.pt'))
head2.load_state_dict(torch.load('head2.pt'))
head3.load_state_dict(torch.load('head3.pt'))
_,predictions,actual=test(test_dataloader)	
print(confusion_matrix(actual,predictions))    	
print(accuracy_score(actual,predictions))
print(classification_report(actual,predictions))
# # Load Saved Model
print("Execution Time --- %s seconds ---" % (time.time() - start_time))
# In[24]:
train_acc=pd.DataFrame(train_acc)
val_acc=pd.DataFrame(val_acc)
train_acc.to_csv("train_acc_bert_base.csv")
val_acc.to_csv('val_acc_bert_base.csv')
train_loss=pd.DataFrame(train_losses)
val_loss=pd.DataFrame(valid_losses)
train_loss.to_csv("train_loss_bert_base.csv")
val_loss.to_csv('val_loss_bert_base.csv')

#load weights of best model
path = 'saved_weights_bert_base.pt'
model.load_state_dict(torch.load(path))
"""
train_acc=pd.DataFrame(train_acc)
val_acc=pd.DataFrame(val_acc)
train_acc.to_csv("bert_base_train_acc.csv",index=False)
val_acc.to_csv("bert_base_val_acc.csv",index=False)
"""
# # Get Predictions for Test Data

# In[25]:


# get predictions for test data
with torch.no_grad():
  preds = model(test_seq.to(device), test_mask.to(device))
  preds = preds.detach().cpu().numpy()


# In[26]:


# model's performance
preds = np.argmax(preds, axis = 1)
print(classification_report(test_y, preds,digits=4))
print(accuracy_score(test_y, preds))
print(confusion_matrix(test_y, preds))

# In[27]:


# confusion matrix
pd.crosstab(test_y, preds)


# In[ ]:




