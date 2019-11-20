import os
import time
import tensorflow as tf
import numpy as np
from base_model import Model
from params_init import random_uniform_initializer, random_normal_initializer, xavier_initializer
from utils.general_utils import Progbar
from utils.general_utils import get_minibatches
from utils.feature_extraction import load_datasets, DataConfig, Flags, punc_pos, pos_prefix
from utils.tf_utils import visualize_sample_embeddings


class ParserModel(Model):
    def __init__(self, config, word_embeddings=None):
        self.word_embeddings = word_embeddings
        self.config = config
        self.build()

    def add_placeholders(self):

        with tf.variable_scope("input_placeholders"):
            self.word_input_placeholder = tf.placeholder(shape=[None, self.config.word_features_types],
                                                         dtype=tf.int32, name="batch_word_indices")
        with tf.variable_scope("label_placeholders"):
            self.labels_placeholder = tf.placeholder(shape=[None, self.config.num_classes],
                                                     dtype=tf.float32, name="batch_one_hot_targets")

    def create_feed_dict(self, inputs_batch, labels_batch=None):
        feed_dict = {
            self.word_input_placeholder: inputs_batch[0]
        }
        if labels_batch is not None:
            feed_dict[self.labels_placeholder] = labels_batch

        return feed_dict


    def write_gradient_summaries(self, grad_tvars):
        with tf.name_scope("gradient_summaries"):
            for (grad, tvar) in grad_tvars:
                mean = tf.reduce_mean(grad)
                stddev = tf.sqrt(tf.reduce_mean(tf.square(grad - mean)))
                tf.summary.histogram("{}/hist".format(tvar.name), grad)
                tf.summary.scalar("{}/mean".format(tvar.name), mean)
                tf.summary.scalar("{}/stddev".format(tvar.name), stddev)
                tf.summary.scalar("{}/sparsity".format(tvar.name), tf.nn.zero_fraction(grad))


    def add_embedding(self):
        with tf.variable_scope("feature_lookup"):
            self.word_embedding_matrix = random_uniform_initializer(#self.word_embeddings.shape, "word_embedding_matrix",
                                                                    [self.config.word_vocab_size, self.config.embedding_dim],
                                                                    "word_embedding_matrix",
                                                                    #0.01, trainable=True)
                                                                    0.01, trainable=False)

            word_context_embeddings = tf.nn.embedding_lookup(self.word_embedding_matrix, self.word_input_placeholder)

            word_embeddings = tf.reshape(word_context_embeddings,
                                         [-1, self.config.word_features_types * self.config.embedding_dim],
                                         name="word_context_embeddings")

        with tf.variable_scope("batch_inputs"):
            embeddings = tf.concat([word_embeddings], 1, name="batch_feature_matrix")

        return embeddings, word_embeddings


    def add_cube_prediction_op(self):
        print("***Building network with CUBE activation***")
        _, word_embeddings = self.add_embedding()

        with tf.variable_scope("layer_connections"):
            with tf.variable_scope("layer_1"):
                w11 = random_uniform_initializer((self.config.word_features_types * self.config.embedding_dim,
                                                  self.config.l1_hidden_size), "w11",
                                                 0.01, trainable=True)
                b1 = random_uniform_initializer((self.config.l1_hidden_size,), "bias1",
                                                0.01, trainable=True)

                # for visualization
                h1 = tf.pow(tf.matmul(word_embeddings, w11) + b1, 3, name="output_activations")

                tf.summary.histogram("output_activations", h1)

            with tf.variable_scope("layer_2"):

                w2 = random_uniform_initializer((self.config.l1_hidden_size, self.config.num_classes), "w2",
                                                0.01, trainable=True)
                b2 = random_uniform_initializer((self.config.num_classes,), "bias2", 0.01, trainable=True)
        with tf.variable_scope("predictions"):
            predictions = tf.add(tf.matmul(h1, w2), b2, name="prediction_logits")

        return predictions


    def add_prediction_op(self):
        pass

    def l2_loss_sum(self, tvars):
        return tf.add_n([tf.nn.l2_loss(t) for t in tvars], "l2_norms_sum")


    def add_loss_op(self, pred):
        tvars = tf.trainable_variables()
        without_bias_tvars = [tvar for tvar in tvars if 'bias' not in tvar.name]

        with tf.variable_scope("loss"):
            cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                labels=self.labels_placeholder, logits=pred), name="batch_xentropy_loss")

            l2_loss = tf.multiply(self.config.reg_val, self.l2_loss_sum(without_bias_tvars), name="l2_loss")
            loss = tf.add(cross_entropy_loss, l2_loss, name="total_batch_loss")
            #loss = cross_entropy_loss

        tf.summary.scalar("batch_loss", loss)

        return loss


    def add_accuracy_op(self, pred):
        with tf.variable_scope("accuracy"):
            accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred, axis=1),
                                                       tf.argmax(self.labels_placeholder, axis=1)), dtype=tf.float32),
                                      name="curr_batch_accuracy")
        return accuracy


    def add_training_op(self, loss):
        with tf.variable_scope("optimizer"):
            #optimizer = tf.train.AdamOptimizer(learning_rate=self.config.lr, name="adam_optimizer")
            optimizer = tf.train.AdagradOptimizer(learning_rate=self.config.lr, name="adagrad_optimizer")
            tvars = tf.trainable_variables()
            grad_tvars = optimizer.compute_gradients(loss, tvars)
            self.write_gradient_summaries(grad_tvars)
            train_op = optimizer.apply_gradients(grad_tvars)

        return train_op

    def train_on_batch(self, sess, inputs_batch, labels_batch, merged):
        word_inputs_batch = inputs_batch
        feed = self.create_feed_dict(word_inputs_batch, labels_batch=labels_batch)
        _, summary, loss = sess.run([self.train_op, merged, self.loss], feed_dict=feed)
        return summary, loss

    def compute_dependencies(self, sess, data, dataset):
        sentences = data
        rem_sentences = [sentence for sentence in sentences]
        [sentence.clear_prediction_dependencies() for sentence in sentences]
        [sentence.clear_children_info() for sentence in sentences]
        dep_offset = (self.config.num_classes-1)/2

        while len(rem_sentences) != 0:
            curr_batch_size = min(dataset.model_config.batch_size, len(rem_sentences))
            batch_sentences = rem_sentences[:curr_batch_size]

            enable_features = [0 if len(sentence.stack) == 1 and len(sentence.buff) == 0 else 1 for sentence in
                               batch_sentences]
            enable_count = np.count_nonzero(enable_features)

            while enable_count > 0:
                curr_sentences = [sentence for i, sentence in enumerate(batch_sentences) if enable_features[i] == 1]

                # get feature for each sentence
                # call predictions -> argmax
                # store dependency and left/right child
                # update state
                # repeat

                curr_inputs = [
                    dataset.feature_extractor.extract_for_current_state(sentence, dataset.word2idx, dataset.dep2idx)[0]
                                                                for sentence in curr_sentences]
                #word_inputs_batch = [curr_inputs[i][0] for i in range(len(curr_inputs))]
                word_inputs_batch = [curr_inputs]
                predictions = sess.run(self.pred,
                                       feed_dict=self.create_feed_dict(word_inputs_batch))
                legal_labels = np.asarray([sentence.get_legal_labels(dep_offset=dep_offset) for sentence in curr_sentences],
                                          dtype=np.float32)
                # crucial: the neural network predicted output is based on
                legal_transitions = np.argmax(predictions + 1000000000 * legal_labels, axis=1)

                # update left/right children so can be used for next feature vector
                [sentence.update_child_dependencies(transition, dep_offset=dep_offset) for (sentence, transition) in
                 zip(curr_sentences, legal_transitions) if transition != self.config.num_classes-1]

                # update state
                [sentence.update_state_by_transition(legal_transition, dep_offset=dep_offset, gold=False) for (sentence, legal_transition) in
                 zip(curr_sentences, legal_transitions)]

                enable_features = [0 if len(sentence.stack) == 1 and len(sentence.buff) == 0 else 1 for sentence in
                                   batch_sentences]
                enable_count = np.count_nonzero(enable_features)

            # Reset stack and buffer
            [sentence.reset_to_initial_state() for sentence in batch_sentences]
            rem_sentences = rem_sentences[curr_batch_size:]


    def get_UAS(self, data, dep2idx=None):
        correct_tokens = 0
        correct_LAS = 0
        all_tokens_LAS = 0
        correct_tokens_total = 0
        all_tokens = 0
        dep_offset = (self.config.num_classes-1)/2
        punc_token_pos = [pos_prefix + each for each in punc_pos]
        for sentence in data:
            # reset each predicted head before evaluation
            [token.reset_predicted_head_id() for token in sentence.tokens]

            head = [-2] * len(sentence.tokens)
            # assert len(sentence.dependencies) == len(sentence.predicted_dependencies)
            for h, t, dep_id in sentence.predicted_dependencies:
                head[t.token_id] = (h.token_id, dep_id%dep_offset)

            non_punc_tokens = [token for token in sentence.tokens if token.pos not in punc_token_pos]
            correct_head_id = [1 if token.head_id == head[token.token_id][0] else 0 for (_, token) in enumerate(
                non_punc_tokens)]
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

    def run_epoch(self, sess, config, dataset, train_writer, merged):
        prog = Progbar(target=1 + len(dataset.train_inputs[0]) / config.batch_size)
        for i, (train_x, train_y) in enumerate(get_minibatches([dataset.train_inputs, dataset.train_targets],
                                                               config.batch_size, is_multi_feature_input=True)):
            # print "input, outout: {}, {}".format(np.array(train_x).shape, np.array(train_y).shape)

            summary, loss = self.train_on_batch(sess, train_x, train_y, merged)
            prog.update(i + 1, [("train loss", loss)])
            # train_writer.add_summary(summary, global_step=i)
        return summary, loss  # Last batch


    def run_valid_epoch(self, sess, dataset):
        print("Evaluating on dev set", end=' ')
        self.compute_dependencies(sess, dataset.valid_data, dataset)
        valid_UAS, valid_LAS, valid_Accu = self.get_UAS(dataset.valid_data, dep2idx=dataset.dep2idx)
        print("- dev Accu: {:.2f}".format(valid_Accu * 100.0))
        print("- dev UAS: {:.2f}".format(valid_UAS * 100.0))
        print("- dev LAS: {:.2f}".format(valid_LAS * 100.0))
        return valid_LAS#valid_UAS


    def fit(self, sess, saver, config, dataset, train_writer, valid_writer, merged):
        best_valid_UAS = 0
        for epoch in range(config.n_epochs):
            print("Epoch {:} out of {:}".format(epoch + 1, self.config.n_epochs))

            summary, loss = self.run_epoch(sess, config, dataset, train_writer, merged)

            if (epoch + 1) % dataset.model_config.run_valid_after_epochs == 0:
                valid_UAS = self.run_valid_epoch(sess, dataset)
                valid_UAS_summary = tf.summary.scalar("valid_UAS", tf.constant(valid_UAS, dtype=tf.float32))
                valid_writer.add_summary(sess.run(valid_UAS_summary), epoch + 1)
                if valid_UAS > best_valid_UAS:
                    best_valid_UAS = valid_UAS
                    if saver:
                        print("New best dev UAS! Saving model..")
                        saver.save(sess, os.path.join(DataConfig.data_dir_path, DataConfig.model_dir,
                                                      DataConfig.model_name))

            # trainable variables summary -> only for training
            if (epoch + 1) % dataset.model_config.write_summary_after_epochs == 0:
                train_writer.add_summary(summary, global_step=epoch + 1)

        print()


def highlight_string(temp):
    print(80 * "=")
    print(temp)
    print(80 * "=")


def main(flag, load_existing_dump=False):
    highlight_string("INITIALIZING")
    print("loading data..")

    dataset = load_datasets(load_existing_dump)
    config = dataset.model_config


if __name__ == '__main__':
    main(Flags.TRAIN, load_existing_dump=False)
