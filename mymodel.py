import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from adaptive import AdaptiveLogSoftmaxWithLoss
import numpy as np
class LSTM(nn.Module):
	def __init__(self, inp_size, hidden_size):
		super().__init__()
		self.inp_size = inp_size
		self.hidden_size = hidden_size
		
		self.i2h = nn.Linear(inp_size, 4 * hidden_size)
		self.h2h = nn.Linear(hidden_size, 4 * hidden_size)
		self.reset_parameters()


	def reset_parameters(self):
		stdv = 1.0 / math.sqrt(self.hidden_size)
		for weight in self.parameters():
			weight.data.uniform_(-stdv, stdv)

	def forward(self, x, hid_state):
		h, c = hid_state
		preact = self.i2h(x) + self.h2h(h)

		gates = preact[:, :3 * self.hidden_size].sigmoid()
		g_t = preact[:, 3 * self.hidden_size:].tanh()
		i_t = gates[:, :self.hidden_size]
		f_t = gates[:, self.hidden_size:2 * self.hidden_size]
		o_t = gates[:, -self.hidden_size:]

		c_t = torch.mul(c, f_t) + torch.mul(i_t, g_t)
		h_t = torch.mul(o_t, c_t.tanh())

		return h_t, (h_t, c_t)                                        
class LSTM_With_H_Detach(nn.Module):
    def __init__(self,input_size,hidden_size,ntoken,vocab_obj,p_detach,criterion):
        super(LSTM_With_H_Detach,self).__init__()
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.ntoken=ntoken
        self.encoder = nn.Embedding(ntoken, input_size)
        embed_matrix_tensor=torch.from_numpy(vocab_obj.embed_matrix).cuda()
        self.encoder.load_state_dict({'weight':embed_matrix_tensor})
        self.p_detach=p_detach
        self.model=LSTM(self.input_size,self.hidden_size).cuda()
        self.criterion=criterion.cuda()
    def  forward(self,x,targets):
        h,c=[],[]
        x=x.long().cuda()

        x=self.encoder(x)
        x=x.transpose(1,0)
           
        h = torch.zeros(x.size(1), self.hidden_size).cuda()
        c = torch.zeros(x.size(1), self.hidden_size).cuda()
        loss=0
        for i in range(x.size(0)):
        	if self.p_detach>0:
        		p_detach = self.p_detach	
        		rand_val = np.random.random(size=1)[0]
        		if rand_val <= p_detach:
        			h = h.detach()
        	output, (h, c) = self.model(inp_x[i], (h, c))
        	loss+=self.criterion(output,targets[i,:])
        loss.backward()

        """for i in range(x.size()[0]):
            curr_ip=x[i,:,:]
            temp_state_h=[]
            temp_state_c=[]
            for j in range(self.num_layers):
                h_t_i,c_t_i=h[i][j],c[i][j]

                c_t,h_t=self.rnn_layers[j](curr_ip,(h_t_i,c_t_i))
                h_t=F.relu(h_t)
                curr_ip=h_t
                if self.dist.sample()==1:
                    temp_state_h.append(h_t.detach())
                else:
                    temp_state_h.append(h_t)
                temp_state_c.append(c_t)
                h_tensor[j][i,:,:]=temp_state_h[-1]
                c_tensor[j][i,:,:]=temp_state_c[-1]"""
            #probs.append(self.decoder(temp_state_h[-1]))
          
        
        
        #out=torch.zeros(len(h)-1,x.size(1),self.ntoken).cuda()
        """for i in range(len(h)-1):
        	#out[i,:,:]=probs[i]
        	for j in range(self.num_layers):
        		h_tensor[j][i,:,:]=h[i+1][j]
        		c_tensor[j][i,:,:]=c[i+1][j]"""

        return output,(h,c),loss
    
    def init_hidden(self,batch_size):
    	h=[torch.zeros(batch_size,self.hidden_size).cuda() for _ in range(self.num_layers)]
    	c=[torch.zeros(batch_size,self.hidden_size).cuda() for _ in range(self.num_layers)]
    	state=[h,c]
    	return state

class lstmmodel(nn.Module):
	def __init__(self,input_size,hidden_size,num_layers,ntoken,vocab_obj):
		super(lstmmodel,self).__init__()
		self.input_size=input_size
		self.hidden_size=hidden_size
		self.num_layers=num_layers
		self.ntoken=ntoken
		self.hidden_state=None
		self.encoder = nn.Embedding(ntoken, input_size)
		embed_matrix_tensor=torch.from_numpy(vocab_obj.embed_matrix).cuda()
		self.encoder.load_state_dict({'weight':embed_matrix_tensor})
		self.lstm=nn.LSTM(self.input_size,int(self.hidden_size/2),self.num_layers,bidirectional=True)
		#self.decoder=nn.Linear(self.hidden_size,ntoken)
		#self.dist = torch.distributions.Bernoulli(torch.Tensor([0.25]))
		#self.rnn_layers.append(LSTM_With_H_Detach_Cell(input_size,hidden_size).cuda())
		#for i in range(num_layers-1):
		#    self.rnn_layers.append(LSTM_With_H_Detach_Cell(hidden_size,hidden_size).cuda())
	def  forward(self,x):
		h,c=[],[]
		x=x.long().cuda()

		x=F.relu(self.encoder(x))
		x=x.transpose(1,0)
		
		x,hidden_size=self.lstm(x,self.hidden)
		return F.relu(x),hidden_size
    
	def init_hidden(self,batch_size):
		return (torch.zeros(2*self.num_layers,batch_size,int(self.hidden_size/2)).cuda(),torch.zeros(2*self.num_layers,batch_size,int(self.hidden_size/2)).cuda())

if __name__=="__main__":
	rnn=LSTM_With_H_Detach(12,16,3)
	h=[torch.zeros(2,16),torch.zeros(2,16),torch.zeros(2,16)]
	c=[torch.zeros(2,16),torch.zeros(2,16),torch.zeros(2,16)]
	state=[h,c]
	ip=torch.zeros(20,2,12)
	op=rnn(ip,state)
	print(op[0][0].size())
	print(op[1][0].size())
