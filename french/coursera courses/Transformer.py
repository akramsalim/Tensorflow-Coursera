import math

import torch
from torch import nn

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(device)
torch.cuda.empty_cache()

num_heads = 8
embed_len = 512
batch_size = 8              # chosen batch size
stack_len = 6               # length of encoder and decoder stacks (=6 as used in paper)
dropout = 0.1               # dropout value to use

output_vocab_size = 7000    # just a dummy number
input_vocab_size = 7000     # just a dummy number


class InputEmbedding(nn.Module):
    def __init__(self, input_vocab_size=input_vocab_size, embed_len=embed_len, dropout=dropout, device=device):
        super(InputEmbedding, self).__init__()
        self.input_vocab_size = input_vocab_size
        self.embed_len = embed_len
        self.device = device
        self.dropout = dropout

        self.firstEmbedding = nn.Embedding(self.input_vocab_size, self.embed_len)
        self.secondEmbedding = nn.Embedding(self.input_vocab_size, self.embed_len) # you need to implement the positional embedding in other way

        self.dropoutLayer = nn.Dropout(p=self.dropout)

    def forward(self, input):
        first_embedding = self.firstEmbedding(input)
        
        batch_size, seq_len = input.shape

        positions_vector = torch.arange(0, seq_len).expand(batch_size, seq_len).to(self.device)
        second_embedding = self.secondEmbedding(positions_vector)

        return self.dropoutLayer(first_embedding + second_embedding)


class ScaledDotProduct(nn.Module):
    def __init__(self, embed_len=embed_len, mask=None):
        super(ScaledDotProduct, self).__init__()
        
        self.dk = embed_len                 # dk = embed_len
        self.mask = mask
        self.softmax = nn.Softmax(dim=3)    # Softmax operator

    # Define the forward function
    def forward(self, queries, keys, values):       

        # First batch MatMul operation & scaling down by sqrt(dk).
        # Output 'compatibility' has shape:
        # (batch_size, num_heads, seq_len, seq_len)
        compatibility = torch.matmul(queries, torch.transpose(keys, 2, 3)) 
        compatibility = compatibility / math.sqrt((self.dk))               

        # Apply mask after scaling the result of MatMul of Q and K.
        # This is needed in the decoder to prevent the decoder from
        # 'peaking ahead' and knowing what word will come next.
        if self.mask is not None:
            compatibility = torch.tril(compatibility)
            
        # Normalize using Softmax
        compatibility_softmax = self.softmax(compatibility)        
               
        return torch.matmul(compatibility_softmax, torch.transpose(values, 1, 2))


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads=num_heads, embed_len=embed_len, batch_size=batch_size, mask=None):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.batch_size = batch_size
        self.embed_len = embed_len
        self.head_length = int(self.embed_len/self.num_heads)
        self.mask = mask
        self.concat_output = []

        # Q, K, and V have shape: (batch_size, seq_len, embed_len)
        self.q_in = self.k_in = self.v_in = self.embed_len

        # Linear layers take in embed_len as input 
        # dim and produce embed_len as output dim
        self.q_linear = nn.Linear(int(self.q_in), int(self.q_in))
        self.k_linear = nn.Linear(int(self.k_in), int(self.k_in))
        self.v_linear = nn.Linear(int(self.v_in), int(self.v_in))

        # Attention layer.
        if self.mask is not None:
            self.attention = ScaledDotProduct(mask=True) 
        else:
            self.attention = ScaledDotProduct()

        self.output_linear = nn.Linear(self.q_in, self.embed_len)

    def forward(self, queries, keys, values):

        # Query has shape: (batch_size, seq_len, num_heads, head_length)
        # Then transpose it: (batch_size, num_heads, seq_len, head_length)
        queries = self.q_linear(queries).reshape(
            self.batch_size, -1, self.num_heads, self.head_length)
        queries = queries.transpose(1, 2)

        # Same for Key as for Query above.
        keys = self.k_linear(keys).reshape(
            self.batch_size, -1, self.num_heads, self.head_length)
        keys = keys.transpose(1, 2)

        # Value has shape: (batch_size, seq_len, num_heads, head_length)
        values = self.v_linear(values).reshape(
            self.batch_size, -1, self.num_heads, self.head_length)

        # 'sdp_output' here has size: 
        # (batch_size, num_heads, seq_len, head_length)
        sdp_output = self.attention.forward(queries, keys, values)

        # Reshape to (batch_size, seq_len, num_heads*head_length)
        sdp_output = sdp_output.transpose(1, 2).reshape(
            self.batch_size, -1, self.num_heads * self.head_length)

        # Return self.output_linear(sdp_output).
        # This has shape (batch_size, seq_len, embed_len)
        return self.output_linear(sdp_output)


class EncoderBlock(nn.Module):
    def __init__(self, embed_len=embed_len, dropout=dropout):
        super(EncoderBlock, self).__init__()

        self.embed_len = embed_len
        self.dropout = dropout
        self.multihead = MultiHeadAttention()             # Multi-Head Attention layer
        self.firstNorm = nn.LayerNorm(embed_len)          # Normalization layer (after the multi-head attention layer)
        self.secondNorm = nn.LayerNorm(embed_len)         # Normalization layer (after the Feed Forward layer)
        self.dropoutLayer = nn.Dropout(p=self.dropout)    # Dropout layer (before addition and normalization)

        # The Feed Forward layer. In the paper this has input &
        # output = 512 (or = embed_len) and inner-layer = 2048 (or = embed_len*4)
        self.feedForward = nn.Sequential(
            nn.Linear(embed_len, embed_len*4),
            nn.ReLU(),
            nn.Linear(embed_len*4, embed_len)
        )

    def forward(self, queries, keys, values):
        attention_output = self.multihead.forward(queries, keys, values)
        attention_output = self.dropoutLayer(attention_output)

        # the output of the first residual connection
        first_sublayer_output = self.firstNorm(attention_output + queries)

        ff_output = self.feedForward(first_sublayer_output)
        ff_output = self.dropoutLayer(ff_output)

        # return the output of the second residual connection
        return self.secondNorm(ff_output + first_sublayer_output)

class DecoderBlock(nn.Module):
    def __init__(self, embed_len=embed_len, dropout=dropout):
        super(DecoderBlock, self).__init__()

        self.embed_len = embed_len
        self.dropout = dropout

        # Masked Multi-Head Attention and Normalization layers.
        self.maskedMultihead = MultiHeadAttention(mask=True)
        self.firstNorm = nn.LayerNorm(self.embed_len)

        self.dropoutLayer = nn.Dropout(p=self.dropout)

        # The output of the above two layers and the output from the encoder stack feed 
        # into an 'encoder block'
        self.encoderBlock = EncoderBlock()

    def forward(self, queries, keys, values):

        # First sublayer, which consists of the Masked Multi-Head Attention + Normalization
        # sublayer, with a residual connection
        masked_multihead_output = self.maskedMultihead.forward(queries, queries, queries)
        masked_multihead_output = self.dropoutLayer(masked_multihead_output)
        first_sublayer_output = self.firstNorm(masked_multihead_output + queries)

        # The remaining of the DecoderBlock is basically an encoder block, which takes keys 
        # and values from the actual Encoder stack output, and takes queries from the 
        # previous sublayer of the DecoderBlock
        return self.encoderBlock.forward(first_sublayer_output, keys, values)      

class Transformer(nn.Module):
    def __init__(self, stack_len=stack_len, embed_len=embed_len, device=device, output_vocab_size=output_vocab_size):
        super(Transformer, self).__init__()
        self.stack_len = stack_len
        self.embed_len = embed_len
        self.device = device
        self.output_vocab_size = output_vocab_size

        self.embedding = InputEmbedding().to(self.device)
        self.encStack = nn.ModuleList([EncoderBlock() for i in range(self.stack_len)])
        self.decStack = nn.ModuleList([DecoderBlock() for i in range(self.stack_len)])
        self.finalLinear = nn.Linear(self.embed_len, self.output_vocab_size)
        self.softmax = nn.Softmax()

    def forward(self, test_input, test_target):

        enc_output = self.embedding.forward(test_input)

        # Final output 'enc_output' of this loop will be both the key and value
        # that will be taken as input to the second sub-layer of the decoder
        for enc_layer in self.encStack:
            enc_output = enc_layer.forward(enc_output, enc_output, enc_output)

        # Decoder stack will take the 'enc_output' from the decoder as the keys
        # and values, and will take its own output from the previous layer as
        # the query. The query used for the first layer is the '<sos>' token.
        dec_output = self.embedding(test_target)
        for dec_layer in self.decStack:
            dec_output = dec_layer.forward(dec_output, enc_output, enc_output)

        # Pass the final decoder stack output to the linear layer that takes in
        # input vector of size 'embed_len' and outputs a vector that has the 
        # size of the vocab specified. Finall return the softmax output of that vector
        final_output = self.finalLinear(dec_output)

        return self.softmax(final_output)
input_tokens = torch.randint(10, (batch_size, 30)).to(device)
output_target = torch.randint(10, (batch_size, 20)).to(device)

Embedding = InputEmbedding().to(device)
input_embeddings = Embedding.forward(input_tokens).to(device)

transformer = Transformer().to(device)
print(input_embeddings.shape)

transformer_output = transformer.forward(input_tokens, output_target)


print(transformer_output.size())
