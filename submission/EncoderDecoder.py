import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderCNN(nn.Module):
    def __init__(self, layers, hparams):
        '''
        Args:
            layers: Description of all layers in the Encoder: [(layer_type, {layer_params})]
                - layer types - ['conv1d', 'conv2d', 'maxpool1d', 'maxpool2d', 'avgpool2d', 'avgpool2d', 'linear', 'dropout']
                - layer_params - dict of parameters for the layer

            hparams: Hyperparameters for the model
        '''
        super(EncoderCNN, self).__init__()
        self.hp = hparams
        self.layers = nn.ModuleList()

        for layer_type, layer_params in layers:
            if layer_type == 'conv1d':
                self.layers.append(nn.Conv1d(**layer_params))
            elif layer_type == 'conv2d':
                self.layers.append(nn.Conv2d(**layer_params))
            elif layer_type == 'maxpool1d':
                self.layers.append(nn.MaxPool1d(**layer_params))
            elif layer_type == 'maxpool2d':
                self.layers.append(nn.MaxPool2d(**layer_params))
            elif layer_type == 'avgpool1d':
                self.layers.append(nn.AvgPool1d(**layer_params))
            elif layer_type == 'avgpool2d':
                self.layers.append(nn.AvgPool2d(**layer_params))
            elif layer_type == 'linear':
                self.layers.append(nn.Linear(**layer_params))
            elif layer_type == 'dropout':
                self.layers.append(nn.Dropout(**layer_params))
            else:
                raise ValueError(f'Invalid layer type: {layer_type}')

    def forward(self, input):
        for layer in self.layers:
            input = layer(input)
        return input
    
class DecoderRNN(nn.Module):
    def __init__(self, vocab, vocab_dict, input_size, embedding_size):
        super(DecoderRNN, self).__init__()
        '''
        Args:
            vocabulary_size: Size of the vocabulary
            embedding_size: Size of the embedding vector
        '''

        self.vocab = vocab
        self.vocab_dict = vocab_dict

        self.embedding = nn.Embedding(len(vocab), embedding_size)
        self.embedding_size = embedding_size
        self.lstm = nn.LSTM(input_size+embedding_size, embedding_size, batch_first=True)
        self.output = nn.Linear(embedding_size, len(vocab))

    def forward(self, input, hidden):
        '''
        Args:
            input: Input to the decoder
            hidden: Hidden state of the previous time step
        '''
        # prev_embed = self.embedding(prev_tokens)
        # concated_inp = torch.cat((input, prev_embed), dim=1)
        if hidden is None:
            output, hidden = self.lstm(input)
        else:
            output, hidden = self.lstm(input, hidden)
        output = self.output(output)

        return output, hidden
    
class EncoderDecoder(nn.Module):
    def __init__(self, encoder=None, decoder=None, device = None):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.device = device
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder.cuda()
        self.decoder = decoder.cuda()

    def forward(self, input):
        context_vec = self.encoder(input).squeeze().to(self.device)
        prev_token = torch.tensor([self.decoder.vocab_dict["<sos>"]], device=self.device)

        input = torch.cat([context_vec.unsqueeze(0).to(self.device), self.decoder.embedding(prev_token).to(self.device)], dim=1).to(self.device)
        hidden = None

        outputs = []

        for i in range(629):
            output, hidden = self.decoder(input, hidden)
            prev_token = torch.argmax(output, dim=1)

            if prev_token.item() == self.decoder.vocab_dict["<eos>"]:
                break
            
            outputs.append(self.decoder.vocab[prev_token.item()])
            input = torch.cat((context_vec.unsqueeze(0), self.decoder.embedding(prev_token)), dim=1).to(self.device)

        return outputs