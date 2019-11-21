import os
import pickle
import argparse
import pandas as pd
import pdb
import functools
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

np.random.seed(1234)
torch.manual_seed(1234)

def read_file(filename, vocab):
	df = pd.read_csv(filename, sep='\t')

	word2idx = vocab['word2idx']
	pos2idx = vocab['pos2idx']
	dep2idx = vocab['dep2idx']
	n_dep = len(dep2idx)

	def get_action(word):
		if word == 'shift':
			return 2 * n_dep
		else:
			parts = word.split(':', 1)
			if parts[0] == 'left':
				return vocab['dep2idx'][parts[1]]
			else:
				if parts[0] == 'right':
					return vocab['dep2idx'][parts[1]] + n_dep
				else:
					pdb.set_trace()
					print('Unexpected')

	df_word = df.iloc[:, :18].applymap(word2idx.get)
	df_pos = df.iloc[:, 18:36].applymap(pos2idx.get) + len(word2idx)
	df_dep = df.iloc[:, 36:48].applymap(dep2idx.get) + (len(word2idx)+len(pos2idx))
	df_action = df.iloc[:, 48:].applymap(get_action)

	assert df_word.isnull().values.any() == False
	assert df_pos.isnull().values.any() == False
	assert df_dep.isnull().values.any() == False

	return np.hstack([df_word, df_pos, df_dep]), df_action.values


def initialize_embedding(filename):
	pickle_filename = filename + '.pickle'
	if os.path.exists(pickle_filename):
		print('Loading embedding from cache')
		with open(pickle_filename, 'rb') as fin:
			return pickle.load(fin)
	else:
		word_embedding = {}
		print('Loading embedding from text')
		for line in open(filename, encoding='utf-8'):
			parts = line.rstrip().split(' ')
			word = parts[0]
			embs = [float(x) for x in parts[1:] if x]
			word_embedding[word] = embs
		with open(pickle_filename, 'wb') as fout:
			pickle.dump(word_embedding, fout, pickle.HIGHEST_PROTOCOL)

		return word_embedding

class Net(nn.Module):
	def __init__(self, word_embedding, vocab):
		super(Net, self).__init__()
		self.word_embedding = nn.Parameter(torch.from_numpy(word_embedding))
		self.fc1 = nn.Linear(50*48, 200)
		self.fc2 = nn.Linear(200, 2*len(vocab['dep2idx']) + 1)

	def forward(self, x):
		emb = F.embedding(x, self.word_embedding).view(-1, 50*48)
		hidden = torch.tanh(self.fc1(emb))
		output = self.fc2(hidden)
		return output

if __name__ == '__main__':
	parser = argparse.ArgumentParser('Nonlinear text classification trainer')
	parser.add_argument('-train', help='train filename', type=str, required=True)
	parser.add_argument('-dev', help='dev filename', type=str, required=True)
	parser.add_argument('-o', help='output filename', type=str, required=True)
	parser.add_argument('-vocab', help='vocab', type=str)
	args = parser.parse_args()
	print('Args:', args)

	with open(args.vocab, 'rb') as fin:
		vocab = pickle.load(fin)

	df_input, df_action = read_file(args.train, vocab)
	df_input_v, df_action_v = read_file(args.dev, vocab)
	word2idx = vocab['word2idx']
	pos2idx = vocab['pos2idx']
	dep2idx = vocab['dep2idx']

	#concat in word, pos, dep order
	word_embedding = np.random.uniform(low=-math.sqrt(3), high=math.sqrt(3), size=(len(word2idx) + len(pos2idx) + len(dep2idx), 50)).astype(np.float32)
	initial_embedding = initialize_embedding('glove.6B.50d.txt')
	unk = initialize_embedding('unk.vec')['UNK']
	word_embedding[word2idx['<unk>']] = unk
	for word, emb in initial_embedding.items():
		idx = word2idx.get(word, -1)
		if idx != -1:
			word_embedding[idx] = emb

	net = Net(word_embedding, vocab)
	net.cuda()
	#net.load_state_dict(torch.load(args.o))

	X = torch.from_numpy(df_input)
	Y = torch.from_numpy(df_action.squeeze())

	X_v = torch.from_numpy(df_input_v).cuda()
	Y_v = torch.from_numpy(df_action_v.squeeze()).cuda()

	dataset = torch.utils.data.TensorDataset(X, Y)
	loader = torch.utils.data.DataLoader(dataset, batch_size=20480, shuffle=True, num_workers=0)
	loss = nn.CrossEntropyLoss()
	opt = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-8)
	#lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.95, patience=2, verbose=True)

	for epoch in range(20):
		for X_batch, Y_batch in loader:
			X_output = net(X_batch.cuda())
			nll = loss(X_output, Y_batch.cuda())
			print(epoch, nll)
			opt.zero_grad()
			nll.backward()
			opt.step()

		#X_output_v = net(X_v)
		#nll_v = loss(X_output_v, Y_v)
		#lr_scheduler.step(nll_v)
		#print('valid', epoch, nll_v)



	torch.save(net.state_dict(), args.o)

	#pdb.set_trace()
	print('123')
