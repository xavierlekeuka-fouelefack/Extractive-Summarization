# Data Challenge
 
# Bert-LSTM Model

The `Bert_lstm` model is a PyTorch-based neural network that combines the power of pre-trained BERT models with the sequential processing capabilities of LSTM (Long Short-Term Memory) networks.

## Model Architecture

The model consists of three main components:

1. **BERT Encoder**: This is a pre-trained BERT model from Hugging Face's model hub. The specific model used is `"intfloat/e5-base-v2"`. The BERT model is used to encode the input data into a meaningful representation that captures the contextual relationships between words in a text.

2. **LSTM**: This is a standard LSTM network with `batch_first=True`. The LSTM takes the output from the BERT encoder (specifically, the `last_hidden_state`) and processes it sequentially, maintaining an internal state that captures information about past elements in the sequence.

3. **Fully Connected Layer**: The output from the LSTM is then passed through a fully connected (linear) layer that maps the LSTM output to the final output size. In this case, the output size is 2, indicating that this model may be used for a binary classification task.

## Forward Pass

During the forward pass, the model takes in some input `x`, processes it through the BERT encoder, passes the result through the LSTM, and finally passes the LSTM output through the fully connected layer. The output of the fully connected layer is the final output of the model.

## Usage
You just need to make a run all to launch all the notebook cells in **Main_model_data_challenge.ipynb** (full pipeline)

Put all the data in the folder so that you will have **test** and **training** folder plus **training_labels.json** .

The paths are already well defined on the notebook.

Here is a basic example of how to initialize and use the model:

```python
from torch import nn
from transformers import AutoModel

class Bert_lstm(nn.Module):
  def __init__(self, hidden_size, output_size):
    super(Bert_lstm, self).__init__()
    self.bert_encoder = AutoModel.from_pretrained("intfloat/e5-base-v2")
    self.hidden_size = hidden_size
    self.lstm = nn.LSTM(self.bert_encoder.config.hidden_size, self.hidden_size, batch_first=True)
    self.fc = nn.Linear(self.hidden_size, 2)

  def forward(self, x):
    bert_outputs = self.bert_encoder(**x)
    last_hidden_states = bert_outputs.last_hidden_state
    outputs, hidden = self.lstm(last_hidden_states)
    out = self.fc(outputs[:, -1, :])
    return out

model = Bert_lstm(256,2).to(device)
print(model) 
