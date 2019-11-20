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
#from torch.utils.data import Dataset, DataLoader
import numpy as np
from preparedata import ModelConfig, DataReader, FeatureExtractor, Dataset
punc_pos = ["''", "``", ":", ".", ","]
pos_prefix = "<p>:"
dep_prefix = "<d>:"

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

def get_UAS(data, dep2idx=None):
	correct_tokens = 0
	correct_LAS = 0
	all_tokens_LAS = 0
	correct_tokens_total = 0
	all_tokens = 0
	dep_offset = len(dep2idx)
	punc_token_pos = [pos_prefix + each for each in punc_pos]
	for sentence in data:
		# reset each predicted head before evaluation
		[token.reset_predicted_head_id() for token in sentence.tokens]

		#pdb.set_trace()
		head = [-2] * len(sentence.tokens)
		# assert len(sentence.dependencies) == len(sentence.predicted_dependencies)
		for h, t, dep_id in sentence.predicted_dependencies:
			head[t.token_id] = (h.token_id, dep_id%dep_offset)

		non_punc_tokens = [token for token in sentence.tokens]
		#non_punc_tokens = [token for token in sentence.tokens if token.pos not in punc_token_pos]
		correct_head_id = [1 if token.head_id == head[token.token_id][0] else 0 for (_, token) in enumerate(non_punc_tokens)]
		#pdb.set_trace()
		res_head_id = sum(correct_head_id)
		correct_tokens += res_head_id

		if dep2idx:
			correct_tokens_total += res_head_id
			correct_dep = [1 if dep2idx[token.dep] == head[token.token_id][1] else 0 for (_, token) in enumerate(non_punc_tokens)]
			correct_tokens_total += sum(correct_dep)

			size = min(len(correct_head_id), len(correct_dep))
			all_tokens_LAS += size*2
			for idx in range(size):
				if correct_head_id[idx] == 1 and correct_dep[idx] == 1:
					correct_LAS += 2

		# all_tokens += len(sentence.tokens)
		all_tokens += len(non_punc_tokens)

	UAS = correct_tokens / float(all_tokens)
	LAS = correct_LAS / float(all_tokens_LAS)
	Accu = correct_tokens_total / float(all_tokens*2)
	return UAS, LAS, Accu

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
	parser.add_argument('-m', help='model filename', type=str, required=True)
	parser.add_argument('-i', help='input filename', type=str, required=True)
	parser.add_argument('-o', help='output filename', type=str, required=True)
	parser.add_argument('-vocab', help='vocab', type=str, required=True)
	args = parser.parse_args()
	print('Args:', args)

	model_config = ModelConfig()

	data_reader = DataReader()
	test_lines = open(args.i, "r").readlines()

	# Load data
	test_data = data_reader.read_data(test_lines)
	print ("Loaded Train data")

	feature_extractor = FeatureExtractor(model_config)
	dataset = Dataset(model_config, test_data, None, None, feature_extractor)

	with open(args.vocab, 'rb') as fin:
		vocab = pickle.load(fin)

	for key, value in vocab.items():
		setattr(dataset, key, value)

	sentences = dataset.train_data
	rem_sentences = [sentence for sentence in sentences]
	[sentence.clear_prediction_dependencies() for sentence in sentences]
	[sentence.clear_children_info() for sentence in sentences]

	word2idx = vocab['word2idx']
	pos2idx = vocab['pos2idx']
	dep2idx = vocab['dep2idx']
	idx2dep = vocab['idx2dep']

	dep_offset = len(dep2idx)
	batch_size = 4096

	word_embedding = np.random.uniform(low=-math.sqrt(3), high=math.sqrt(3), size=(len(word2idx) + len(pos2idx) + len(dep2idx), 50)).astype(np.float32)
	net = Net(word_embedding, vocab)
	net.load_state_dict(torch.load(args.m))
	while len(rem_sentences) != 0:
		curr_batch_size = min(batch_size, len(rem_sentences))
		batch_sentences = rem_sentences[:curr_batch_size]

		enable_features = [0 if len(sentence.stack) == 1 and len(sentence.buff) == 0 else 1 for sentence in batch_sentences]
		enable_count = np.count_nonzero(enable_features)

		while enable_count > 0:
			curr_sentences = [sentence for i, sentence in enumerate(batch_sentences) if enable_features[i] == 1]

			# get feature for each sentence
			# call predictions -> argmax
			# store dependency and left/right child
			# update state
			# repeat

			curr_inputs = [
				dataset.feature_extractor.extract_for_current_state(sentence, dataset.word2idx, dataset.pos2idx, dataset.dep2idx, to_index=True)
															for sentence in curr_sentences]

			word_inputs_batch = np.asarray([x[0] for x in curr_inputs])
			pos_inputs_batch = np.asarray([x[1] for x in curr_inputs]) + len(word2idx)
			dep_inputs_batch = np.asarray([x[2] for x in curr_inputs]) + (len(word2idx)+len(pos2idx))

			inputs = torch.from_numpy(np.hstack([word_inputs_batch, pos_inputs_batch, dep_inputs_batch])).type(torch.LongTensor)
			predictions = net(inputs).detach().numpy()
			#pdb.set_trace()

			legal_labels = np.asarray([sentence.get_legal_labels(dep_offset=dep_offset) for sentence in curr_sentences], dtype=np.float32)
			# crucial: the neural network predicted output is based on
			#pdb.set_trace()
			legal_transitions = np.argmax(predictions + 1000000000 * legal_labels, axis=1)

			# update left/right children so can be used for next feature vector
			[sentence.update_child_dependencies(transition, dep_offset=dep_offset) for (sentence, transition) in zip(curr_sentences, legal_transitions) if transition != (dep_offset*2)]

			# update state
			[sentence.update_state_by_transition(legal_transition, dep_offset=dep_offset, gold=False) for (sentence, legal_transition) in zip(curr_sentences, legal_transitions)]

			enable_features = [0 if len(sentence.stack) == 1 and len(sentence.buff) == 0 else 1 for sentence in batch_sentences]
			enable_count = np.count_nonzero(enable_features)

		# Reset stack and buffer
		[sentence.reset_to_initial_state() for sentence in batch_sentences]
		rem_sentences = rem_sentences[curr_batch_size:]

	valid_UAS, valid_LAS, valid_Accu = get_UAS(dataset.train_data, dep2idx=dataset.dep2idx)
	#pdb.set_trace()
	with open(args.o, 'w') as fout:
		for sentence in dataset.train_data:
			head = [-2] * len(sentence.tokens)
			for h, t, dep_id in sentence.predicted_dependencies:
				head[t.token_id] = (h.token_id, dep_id%dep_offset)

			for i, token in enumerate(sentence.tokens):
				fields = token.line.rstrip().split('\t')
				#pdb.set_trace()
				fields[6] = str(head[i][0] + 1)
				fields[7] = idx2dep[head[i][1]].replace('<d>:', '')
				fout.write('\t'.join(fields) + '\n')
			fout.write('\n')

	pdb.set_trace()
	print('123')
