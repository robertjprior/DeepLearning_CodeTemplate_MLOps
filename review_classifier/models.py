#setup model architecture
from transformers import RobertaModel
import torch
from torch import nn
from config.config import NUM_LABELS, DROPOUT

class RobertaClassifier(nn.Module):
    def __init__(self, num_labels=NUM_LABELS, dropout=DROPOUT, averaging = None):
        super().__init__()

        self.num_labels = num_labels
        self.averaging = averaging

        self.dropout = nn.Dropout(dropout)
        self.bert = RobertaModel.from_pretrained("roberta-base", return_dict=True)
        self.hidden_size = self.bert.config.hidden_size
        
        self.dense = nn.Linear(self.hidden_size, self.hidden_size) #https://github.com/google-research/bert/issues/43
        #https://discuss.huggingface.co/t/what-is-the-purpose-of-the-additional-dense-layer-in-classification-heads/526
        self.linear = nn.Linear(self.hidden_size, num_labels)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
    
    def forward(self, input_ids, attention_mask):
        
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, return_dict=True)
        if self.averaging=="last":
            #average across this middle 
            sentence_representation = torch.mean(outputs['last_hidden_state'], 1)
            #TODO: should ultimately try to avoid padding tokens https://stackoverflow.com/questions/71434804/how-to-fed-last-4-concatenated-hidden-layers-of-bert-to-fc-layers
        if self.averaging == "last four":
            feature_layers = outputs['hidden_states'][-4:]
            sentence_representation = torch.cat(feature_layers, -1) #concatenate them (here over the last dimension) to a single tensor of shape (batch_size, seq_len, 4 * hidden_size)
            sentence_representation = torch.mean(sentence_representation, 1)
            self.hidden_size = self.hidden_size *4
            self.dense = nn.Linear(self.hidden_size, self.hidden_size) #https://github.com/google-research/bert/issues/43
            self.linear = nn.Linear(self.hidden_size, self.num_labels)

        else: #if none
            sentence_representation = outputs['last_hidden_state'][:, 0, :] #cls token
        x = self.dropout(sentence_representation)
        x = self.dense(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear(x)
        x = self.softmax(x)
        return x
        
