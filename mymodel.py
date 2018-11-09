import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM_With_H_Detach_Cell(nn.Module):
    def __init__(self,input_size,hidden_size):
        super(LSTM_With_H_Detach_Cell,self).__init__()
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.f=nn.Linear(input_size+hidden_size,hidden_size)
        self.i=nn.Linear(input_size+hidden_size,hidden_size)
        self.c=nn.Linear(input_size+hidden_size,hidden_size)
        self.o=nn.Linear(input_size+hidden_size,hidden_size)
    def forward(self,x,state):
        h,c=state[0],state[1]
        ip=torch.cat((x,h),dim=1)
        gatef=F.sigmoid(self.f(ip))
        gatei=F.sigmoid(self.i(ip))
        gatec=F.tanh(self.c(ip))
        gateo=F.sigmoid(self.o(ip))
        c=gatef*c+gatei*gatec   
        h=gateo*F.tanh(c)
        return c,h

class LSTM_With_H_Detach(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers,ntoken,vocab_obj):
        super(LSTM_With_H_Detach,self).__init__()
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.num_layers=num_layers
        self.rnn_layers=[]
        self.ntoken=ntoken
        self.encoder = nn.Embedding(ntoken, input_size)
        embed_matrix_tensor=torch.from_numpy(vocab_obj.embed_matrix).cuda()
        self.encoder.load_state_dict({'weight':embed_matrix_tensor})
        #self.decoder=nn.Linear(self.hidden_size,ntoken)
        self.dist = torch.distributions.Bernoulli(torch.Tensor([0.25]))
        self.rnn_layers.append(LSTM_With_H_Detach_Cell(input_size,hidden_size).cuda())
        for i in range(num_layers-1):
            self.rnn_layers.append(LSTM_With_H_Detach_Cell(hidden_size,hidden_size).cuda())
    def  forward(self,x,state,eval):
        h,c=[],[]
        x=x.long().cuda()

        x=self.encoder(x)
        x=x.transpose(1,0)
        h_tensor,c_tensor=[],[]
           
        h_tensor=[torch.zeros(x.size()[0],x.size()[1],self.hidden_size).cuda() for _ in range(self.num_layers)]
        c_tensor=[torch.zeros(x.size()[0],x.size()[1],self.hidden_size).cuda() for _ in range(self.num_layers)]
        h.append(state[0])
        c.append(state[1])
        probs=[]
        for i in range(x.size()[0]):
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
                c_tensor[j][i,:,:]=temp_state_c[-1]
            #probs.append(self.decoder(temp_state_h[-1]))
            if eval==1:
            	if i+1<x.size(0):
            		x[i+1,0,:]=self.encoder(torch.argmax(probs[i].view(-1)).view(1,1))
            h.append(temp_state_h)
            c.append(temp_state_c)
        
        
        #out=torch.zeros(len(h)-1,x.size(1),self.ntoken).cuda()
        """for i in range(len(h)-1):
        	#out[i,:,:]=probs[i]
        	for j in range(self.num_layers):
        		h_tensor[j][i,:,:]=h[i+1][j]
        		c_tensor[j][i,:,:]=c[i+1][j]"""

        return h_tensor,c_tensor,None
    
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

		x=self.encoder(x)
		x=x.transpose(1,0)
		hidden_size=self.init_hidden(x.size(1))
		x,hidden_size=self.lstm(x,hidden_size)
		return x
    
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
