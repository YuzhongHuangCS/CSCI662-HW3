import os
import numpy as np
import datetime
from enum import Enum
import argparse
import pdb
import pickle
import copy

NULL = "<null>"
UNK = "<unk>"
ROOT = "<root>"
pos_prefix = "<p>:"
dep_prefix = "<d>:"
punc_pos = ["''", "``", ":", ".", ","]

today_date = str(datetime.datetime.now().date())

def list_to_vocab(items):
	return {value: index for index, value in enumerate(items)}

class Token(object):
	def __init__(self, token_index, word, pos, dep, head_index, line=None):
		super(Token, self).__init__()

		self.token_index = token_index  # token index
		self.word = word.lower()
		self.pos = pos_prefix + pos
		self.dep = dep_prefix + dep
		self.head_index = head_index  # head token index
		self.left_children = list()
		self.right_children = list()
		self.line = line

	def is_root_token(self):
		return self.word == ROOT

	def is_null_token(self):
		return self.word == NULL

	def is_unk_token(self):
		return self.word == UNK

NULL_TOKEN = Token(-1, NULL, NULL, NULL, -1)
ROOT_TOKEN = Token(-1, ROOT, ROOT, ROOT, -1)
UNK_TOKEN = Token(-1, UNK, UNK, UNK, -1)


class Sentence(object):
	def __init__(self, tokens):
		self.root = copy.deepcopy(ROOT_TOKEN)
		self.tokens = tokens
		self.buff = [token for token in self.tokens]
		self.stack = [self.root]
		self.dependencies = []
		self.predicted_dependencies = []

	def update_child_dependencies(self, curr_transition, dep_offset=1):
		if curr_transition >= 0 and curr_transition < dep_offset:
			head = self.stack[-1]
			dependent = self.stack[-2]
		elif curr_transition >= dep_offset and curr_transition < 2*dep_offset:
			head = self.stack[-2]
			dependent = self.stack[-1]

		if head.token_index > dependent.token_index:
			head.left_children.append(dependent.token_index)
			head.left_children.sort()
		else:
			head.right_children.append(dependent.token_index)
			head.right_children.sort()

	def get_child_by_index_and_depth(self, token, index, direction, depth):  # Get child token
		if depth == 0:
			return token

		if direction == "left":
			if len(token.left_children) > index:
				return self.get_child_by_index_and_depth(self.tokens[token.left_children[index]], index, direction, depth - 1)
			return NULL_TOKEN
		else:
			if len(token.right_children) > index:
				return self.get_child_by_index_and_depth(self.tokens[token.right_children[::-1][index]], index, direction, depth - 1)
			return NULL_TOKEN


	def get_legal_labels(self, dep_offset=1):
		labels = ([1] if len(self.stack) > 2 else [0]) * int(dep_offset)
		labels += ([1] if len(self.stack) >= 2 else [0]) * int(dep_offset)
		labels += [1] if len(self.buff) > 0 else [0]
		return labels


	# return -1: not projective
	def get_transition_from_current_state(self, dep2idx=None, dep_offset=1):  # logic to get next transition
		if len(self.stack) < 2:# shift
			return 2*dep_offset

		stack_token_0 = self.stack[-1]
		stack_token_1 = self.stack[-2]
		if stack_token_1.token_index >= 0 and stack_token_1.head_index == stack_token_0.token_index:  # left arc
			return 0 if not dep2idx else dep2idx[stack_token_1.dep]
		elif stack_token_1.token_index >= -1 and stack_token_0.head_index == stack_token_1.token_index and stack_token_0.token_index not in [x.head_index for x in self.buff]:# right arc
			return 1 if not dep2idx else dep_offset+dep2idx[stack_token_0.dep]
		else:
			return 2*dep_offset if len(self.buff) != 0 else None


	# [0, dep_offset): left arc
	# [dep_offset, 2*dep_offset): right arc
	# 2*dep_offset: shift
	def update_state_by_transition(self, transition, dep_offset=1, gold=True):  # updates stack, buffer and dependencies
		if transition is not None:
			if transition == 2*dep_offset:  # shift
				self.stack.append(self.buff[0])
				self.buff = self.buff[1:] if len(self.buff) > 1 else []
			elif transition >= 0 and transition < dep_offset:  # left arc
				self.dependencies.append(
					(self.stack[-1], self.stack[-2], transition)) if gold else self.predicted_dependencies.append(
					(self.stack[-1], self.stack[-2], transition))
				#left arc: head: -1
				self.stack = self.stack[:-2] + self.stack[-1:]
			elif transition >= dep_offset and transition < 2*dep_offset:  # right arc
				self.dependencies.append(
					(self.stack[-2], self.stack[-1], transition)) if gold else self.predicted_dependencies.append(
					(self.stack[-2], self.stack[-1], transition))
				#right arc: head: -2
				self.stack = self.stack[:-1]


	def reset_to_initial_state(self):
		self.buff = [token for token in self.tokens]
		self.stack = [self.root]


	def clear_prediction_dependencies(self):
		self.predicted_dependencies = []


	def clear_children_info(self):
		for token in self.tokens:
			token.left_children = []
			token.right_children = []


class Dataset(object):
	def __init__(self, train_data, feature_extractor):
		self.train_data = train_data
		self.feature_extractor = feature_extractor

		# Vocab
		self.word2idx = None
		self.idx2word = None
		self.dep2idx = None
		self.idx2dep = None

		# input & outputs
		self.train_inputs, self.train_targets = None, None

	def build_vocab(self):
		all_words = set()
		all_pos = set()
		all_dep = set()

		for sentence in self.train_data:
			all_words.update(set([x.word for x in sentence.tokens]))
			all_pos.update(set([x.pos for x in sentence.tokens]))
			all_dep.update(set([x.dep for x in sentence.tokens]))

		all_words.add(ROOT_TOKEN.word)
		all_words.add(NULL_TOKEN.word)
		all_words.add(UNK_TOKEN.word)

		all_pos.add(ROOT_TOKEN.pos)
		all_pos.add(NULL_TOKEN.pos)
		all_pos.add(UNK_TOKEN.pos)

		all_dep.add(ROOT_TOKEN.dep)
		all_dep.add(NULL_TOKEN.dep)
		all_dep.add(UNK_TOKEN.dep)

		word_vocab = list(all_words)
		pos_vocab = list(all_pos)
		dep_vocab = list(all_dep)

		word2idx = list_to_vocab(word_vocab)
		idx2word = {idx: word for (word, idx) in list(word2idx.items())}

		pos2idx = list_to_vocab(pos_vocab)
		idx2pos = {idx: pos for (pos, idx) in list(pos2idx.items())}

		dep2idx = list_to_vocab(dep_vocab)
		idx2dep = {idx: dep for (dep, idx) in list(dep2idx.items())}

		self.word2idx = word2idx
		self.idx2word = idx2word

		self.pos2idx = pos2idx
		self.idx2pos = idx2pos

		self.dep2idx = dep2idx
		self.idx2dep = idx2dep

	def convert_data_to_ids(self):
		self.train_inputs, self.train_targets = self.feature_extractor.create_instances_for_data(self.train_data, self.word2idx, self.pos2idx, self.dep2idx, self.idx2dep)


class FeatureExtractor(object):
	def __init__(self):
		super(FeatureExtractor, self).__init__()

	def extract_from_stack_and_buffer(self, sentence, num_words=3):
		tokens = []

		tokens.extend(list(reversed(sentence.stack[-num_words:])))
		tokens.extend([NULL_TOKEN for _ in range(num_words - len(sentence.stack))])
		#tokens.extend([NULL_TOKEN for _ in range(num_words - len(sentence.stack))])
		#tokens.extend(sentence.stack[-num_words:])

		tokens.extend(sentence.buff[:num_words])
		tokens.extend([NULL_TOKEN for _ in range(num_words - len(sentence.buff))])
		return tokens  # 6 features


	def extract_children_from_stack(self, sentence, num_stack_words=2):
		children_tokens = []

		for i in range(num_stack_words):
			if len(sentence.stack) > i:
				lc0 = sentence.get_child_by_index_and_depth(sentence.stack[-i - 1], 0, "left", 1)
				rc0 = sentence.get_child_by_index_and_depth(sentence.stack[-i - 1], 0, "right", 1)

				lc1 = sentence.get_child_by_index_and_depth(sentence.stack[-i - 1], 1, "left",
															1) if lc0 != NULL_TOKEN else NULL_TOKEN
				rc1 = sentence.get_child_by_index_and_depth(sentence.stack[-i - 1], 1, "right",
															1) if rc0 != NULL_TOKEN else NULL_TOKEN

				llc0 = sentence.get_child_by_index_and_depth(sentence.stack[-i - 1], 0, "left",
															 2) if lc0 != NULL_TOKEN else NULL_TOKEN
				rrc0 = sentence.get_child_by_index_and_depth(sentence.stack[-i - 1], 0, "right",
															 2) if rc0 != NULL_TOKEN else NULL_TOKEN

				children_tokens.extend([lc0, rc0, lc1, rc1, llc0, rrc0])
			else:
				[children_tokens.append(NULL_TOKEN) for _ in range(6)]

		return children_tokens  # 12 features


	def extract_for_current_state(self, sentence, word2idx, pos2idx, dep2idx, to_index=False):
		direct_tokens = self.extract_from_stack_and_buffer(sentence, num_words=3)
		children_tokens = self.extract_children_from_stack(sentence, num_stack_words=2)

		word_features = []
		pos_features = []
		dep_features = []

		# Word features -> 18
		word_features.extend([x.word for x in direct_tokens])
		word_features.extend([x.word for x in children_tokens])

		# pos features -> 18
		pos_features.extend(map(lambda x: x.pos, direct_tokens))
		pos_features.extend(map(lambda x: x.pos, children_tokens))

		# dep features -> 12 (only children)
		#dep_features.extend(map(lambda x: x.dep, direct_tokens))
		dep_features.extend([x.dep for x in children_tokens])

		if to_index:
			word_input_ids = [word2idx[word] if word in word2idx else word2idx[UNK_TOKEN.word] for word in word_features]
			pos_input_ids = [pos2idx[pos] if pos in pos2idx else pos2idx[UNK_TOKEN.pos] for pos in pos_features]
			dep_input_ids = [dep2idx[dep] if dep in dep2idx else dep2idx[UNK_TOKEN.dep] for dep in dep_features]
		else:
			word_input_ids = [word if word in word2idx else UNK_TOKEN.word for word in word_features]
			pos_input_ids = [pos if pos in pos2idx else UNK_TOKEN.pos for pos in pos_features]
			dep_input_ids = [dep if dep in dep2idx else UNK_TOKEN.dep for dep in dep_features]
			#word_input_ids = [word2idx[word] if word in word2idx else word2idx[UNK_TOKEN.word] for word in word_features]
			#dep_input_ids = [dep2idx[dep] if dep in dep2idx else dep2idx[UNK_TOKEN.dep] for dep in dep_features]

		return [word_input_ids, pos_input_ids, dep_input_ids]  # 48 features


	def create_instances_for_data(self, data, word2idx, pos2idx, dep2idx, idx2dep):
		labels = []
		word_inputs = []
		pos_inputs = []
		dep_inputs = []

		def get_dep_from_idx(idx):
			if idx < len(idx2dep):
				return 'left:' + idx2dep[idx]
			else:
				if idx < 2*len(idx2dep):
					return 'right:' + idx2dep[idx-len(idx2dep)]
				else:
					return 'shift'

		non_projective = 0
		for i, sentence in enumerate(data):
			num_words = len(sentence.tokens)
			this_word_inputs = []
			this_pos_inputs = []
			this_dep_inputs = []
			this_labels = []

			for _ in range(num_words * 2):
				word_input, pos_input, dep_input = self.extract_for_current_state(sentence, word2idx, pos2idx, dep2idx, to_index=False)
				legal_labels = sentence.get_legal_labels(dep_offset=len(dep2idx))
				curr_transition = sentence.get_transition_from_current_state(dep2idx=dep2idx,
																			dep_offset=len(dep2idx))
				if curr_transition is None:
					break

				assert legal_labels[curr_transition] == 1

				# updateate left/right children
				if curr_transition != 2*len(dep2idx):
					sentence.update_child_dependencies(curr_transition, dep_offset=len(dep2idx))

				sentence.update_state_by_transition(curr_transition, dep_offset=len(dep2idx))


				this_labels.append(get_dep_from_idx(curr_transition))
				this_word_inputs.append(word_input)
				this_pos_inputs.append(pos_input)
				this_dep_inputs.append(dep_input)
			#else:
			#	sentence.reset_to_initial_state()

			if len(sentence.buff) == 0 and len(sentence.stack) == 1 and sentence.stack[0].word == ROOT_TOKEN.word:
				word_inputs.extend(this_word_inputs)
				pos_inputs.extend(this_pos_inputs)
				dep_inputs.extend(this_dep_inputs)
				labels.extend(this_labels)
			else:
				#pdb.set_trace()
				non_projective += 1
				word_inputs.extend(this_word_inputs)
				pos_inputs.extend(this_pos_inputs)
				dep_inputs.extend(this_dep_inputs)
				labels.extend(this_labels)

			# reset stack and buffer to default state
			sentence.reset_to_initial_state()

		print('Total sentences: {}, non_projective: {}, ratio: {}'.format(len(data), non_projective, non_projective/len(data)))
		print('Total configuration: {}'.format(len(labels)))
		return [word_inputs, pos_inputs, dep_inputs], labels


class DataReader(object):
	def __init__(self):
		super(DataReader, self).__init__()

	def read_conll(self, token_lines):
		tokens = []
		for each in token_lines:
			fields = each.strip().split("\t")
			try:
				token_index = int(fields[0]) - 1
			except:
				break
			word = fields[1]
			pos = fields[4]
			dep = fields[7]
			head_index = int(fields[6]) - 1
			token = Token(token_index, word, pos, dep, head_index, each)
			tokens.append(token)
		sentence = Sentence(tokens)

		return sentence


	def read_data(self, data_lines):
		data_objects = []
		token_lines = []
		for token_conll in data_lines:
			token_conll = token_conll.strip()
			if len(token_conll) > 0:
				if token_conll[0] != '#':
					token_lines.append(token_conll)
			else:
				objects = self.read_conll(token_lines)
				if objects:
					data_objects.append(objects)
				token_lines = []
		if len(token_lines) > 0:
			objects = self.read_conll(token_lines)
			if objects:
				data_objects.append(objects)
		return data_objects

if __name__ == '__main__':
	parser = argparse.ArgumentParser('Nonlinear text classification trainer')
	parser.add_argument('-i', help='input filename', type=str, required=True)
	parser.add_argument('-o', help='output filename', type=str, required=True)
	parser.add_argument('-vocab', help='vocab', type=str)
	args = parser.parse_args()
	print('Args:', args)


	data_reader = DataReader()
	train_lines = open(args.i, "r").readlines()

	# Load data
	train_data = data_reader.read_data(train_lines)
	print ("Loaded Train data")

	feature_extractor = FeatureExtractor()
	dataset = Dataset(train_data, feature_extractor)

	if args.vocab:
		if os.path.exists(args.vocab):
			with open(args.vocab, 'rb') as fin:
				vocab = pickle.load(fin)

			for key, value in vocab.items():
				setattr(dataset, key, value)

			pdb.set_trace()
		else:
			print('Vocab file: {} Not found.'.format(args.vocab))
	else:
		dataset.build_vocab()
		with open('vocab.pickle', 'wb') as fout:
			dump = {
				'word2idx': dataset.word2idx,
				'idx2word': dataset.idx2word,
				'pos2idx': dataset.pos2idx,
				'idx2pos': dataset.pos2idx,
				'dep2idx': dataset.dep2idx,
				'idx2dep': dataset.idx2dep
			}
			pickle.dump(dump, fout, pickle.HIGHEST_PROTOCOL)


	print("Vocab Build Done!")
	#dataset.model_config.word_vocab_size = len(dataset.word2idx)
	print("converting data into ids..")

	dataset.convert_data_to_ids()
	print("Done!")
	#pdb.set_trace()
	#assert len(dataset.train_targets[0]) == dataset.model_config.num_classes
	#dataset.model_config.num_classes = len(dataset.train_targets[0])

	with open(args.o, 'w') as fout:
		columns = 's1:w,s2:w,s3:w,b1:w,b2:w,b3:w,1lc1:w,1rc1:w,1lc2:w,1rc2:w,1llc1:w,1rrc1:w,2lc1:w,2rc1:w,2lc2:w,2rc2:w,2llc1:w,2rrc1:w,'
		columns += 's1:p,s2:p,s3:p,b1:p,b2:p,b3:p,1lc1:p,1rc1:p,1lc2:p,1rc2:p,1llc1:p,1rrc1:p,2lc1:p,2rc1:p,2lc2:p,2rc2:p,2llc1:p,2rrc1:p,'
		columns += '1lc1:d,1rc1:d,1lc2:d,1rc2:d,1llc1:d,1rrc1:d,2lc1:d,2rc1:d,2lc2:d,2rc2:d,2llc1:d,2rrc1:d,'
		columns += 'action\n'
		fout.write(columns.replace(',', '\t'))

		for i in range(len(dataset.train_targets)):
			word_inputs = dataset.train_inputs[0][i]
			pos_inputs = dataset.train_inputs[1][i]
			dep_inputs = dataset.train_inputs[2][i]
			action = dataset.train_targets[i]

			fout.write('\t'.join(word_inputs) + '\t')
			fout.write('\t'.join(pos_inputs) + '\t')
			fout.write('\t'.join(dep_inputs) + '\t')
			fout.write(action + '\n')

	print(len(dataset.train_inputs[0]))
	#pdb.set_trace()
	print(123)
