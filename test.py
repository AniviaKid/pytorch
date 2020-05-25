import torch
import spearman as sp
import scipy.stats as stats
import pandas as pd
import data

device = torch.device("cuda")

def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)

def get_embedding_vector(model,input,a,b):
    return model.encoder(input)

#def get_embedding_vector(net,input_data,seq_length,embed_length):
#  x=net.encoder(input_data)
#  return x

def get_spearman():
    word_data, human = sp.Load_data('./data/wikitext-2/combined.csv')
    #corpus = sp.Corpus(data,human)
    corpus = data.Corpus('./data/wikitext-2')
    #encoder = model.get_embedding()
    model=torch.load('./model.pt')
    #word_data = batchify(corpus.train, 1)
    cosine = []
    delete = []
    for i in range(0,len(word_data),1):
        if word_data[i][0] not in corpus.dictionary.word2idx or word_data[i][1] not in corpus.dictionary.word2idx:
            delete.append(i)
            continue
        a=get_embedding_vector(model,torch.tensor([corpus.dictionary.word2idx[word_data[i][0]]]).type(torch.int64).to(device))
        b=get_embedding_vector(model,torch.tensor([corpus.dictionary.word2idx[word_data[i][1]]]).type(torch.int64).to(device))
        cosine.append(torch.cosine_similarity(a,b).item())
    for i in range(0,len(delete),1):
        human.pop(delete[len(delete)-i-1])

    
    #vector = encoder(word_data) #ntokens*100，included repeated word
    #vector = get_embedding_vector(model,word_data)
    #for i in range(0,vector.size(0)-1,2):
        #cosine.append(torch.cosine_similarity(vector[i][0],vector[i+1][0],dim=-1))
    #f=open("./tmp_output.txt","w+")
    #for i in range(0,len(cosine)-1):
    #    print(cosine[i],file=f)
    #f.close()
    #human_float=[]
    #cosine_float=[]
    #for i in human:
    #    human_float.append(math.tanh(float(i)))
    #for i in cosine:
    #    cosine_float.append(float(i))
    #spearman=stats.spearmanr(cosine_float,human_float)
    spearman=stats.spearmanr(cosine, human)
    print(spearman)




df=pd.read_csv('./data/wikitext-2/combined.csv')
model=torch.load('./model.pt')
corpus = data.Corpus('./data/wikitext-2')
cosine=[]
human=[]
seq_length=10
embed_length=100 
neurons=100 
for i in range(len(df['Word 1'])):
  if df['Word 1'][i] not in corpus.dictionary.word2idx or df['Word 2'][i] not in corpus.dictionary.word2idx:
    continue
  a=get_embedding_vector(model,torch.tensor([corpus.dictionary.word2idx[df['Word 1'][i]]]).type(torch.int64).to(device),seq_length,embed_length)
  b=get_embedding_vector(model,torch.tensor([corpus.dictionary.word2idx[df['Word 2'][i]]]).type(torch.int64).to(device),seq_length,embed_length)
  cosine.append(torch.cosine_similarity(a,b).item())
  human.append(df['Human (mean)'][i])

cosine=pd.DataFrame(cosine,columns=['machine mean'])
human=pd.DataFrame(human,columns=['human mean'])
alldata=pd.concat([cosine,human],axis=1)
print(alldata.corr(method='spearman'))