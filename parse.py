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
from preparedata import DataReader, FeatureExtractor, Dataset
from train import Net

np.random.seed(1234)
torch.manual_seed(1234)

punc_pos = ["''", "``", ":", ".", ","]
pos_prefix = "<p>:"
dep_prefix = "<d>:"

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

if __name__ == '__main__':
	parser = argparse.ArgumentParser('Nonlinear text classification trainer')
	parser.add_argument('-m', help='model filename', type=str, required=True)
	parser.add_argument('-i', help='input filename', type=str, required=True)
	parser.add_argument('-o', help='output filename', type=str, required=True)
	parser.add_argument('-vocab', help='vocab', type=str, required=True)
	args = parser.parse_args()
	print('Args:', args)

	data_reader = DataReader()
	test_lines = open(args.i, "r").readlines()

	# Load data
	test_data = data_reader.read_data(test_lines)
	print ("Loaded Train data")

	feature_extractor = FeatureExtractor()
	dataset = Dataset(test_data, feature_extractor)

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

	with open(args.o, 'w') as fout:
		for sentence in dataset.train_data:
			head = [-2] * len(sentence.tokens)
			for h, t, dep_id in sentence.predicted_dependencies:
				head[t.token_index] = (h.token_index, dep_id%dep_offset)

			for i, token in enumerate(sentence.tokens):
				fields = token.line.rstrip().split('\t')
				#pdb.set_trace()
				fields[6] = str(head[i][0] + 1)
				fields[7] = idx2dep[head[i][1]].replace('<d>:', '')
				fout.write('\t'.join(fields) + '\n')
			fout.write('\n')

	#pdb.set_trace()
