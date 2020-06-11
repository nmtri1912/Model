import time
import math
import pickle
import collections
from nltk import word_tokenize
import nltk
import argparse

import tensorflow as tf
from tensorflow.contrib import rnn
import re
import numpy as np
from gensim.models import KeyedVectors
from sklearn.decomposition import PCA
from matplotlib import pyplot

# argument
args = argparse.ArgumentParser()
args.add_argument("--language_src")
args.add_argument("--language_targ")
args.add_argument("--vocab_src")
args.add_argument("--vocab_targ")
args.add_argument("--word_emb_src")
args.add_argument("--word_emb_targ")
args.add_argument("--num_layers")
args.add_argument("--num_hiddens")
args.add_argument("--learning_rate")
args.add_argument("--keep_prob")
args.add_argument("--beam_width")
args.add_argument("--batch_size")
args.add_argument("--checkpoint")

args = vars(args.parse_args())

# Set paths
train_english_path = args["language_src"]
train_vietnamese_path = args["language_targ"]
word2int_english_path = args["vocab_src"] + "word2int.pickle"
int2word_english_path = args["vocab_src"] + "int2word.pickle"
word2int_vietnamese_path = args["vocab_targ"] + "word2int.pickle"
int2word_vietnamese_path = args["vocab_targ"] + "int2word.pickle"


def getSentList(path, n_sents):
    sentList = []
    with open(path) as f:
        # using 100 to test the code only
        for line in f.readlines()[:n_sents]:
            w = line.lower()
            w = re.sub(r"([?.!,Â¿])", r" \1 ", w)
            w = re.sub(r'[" "]+', " ", w)
            #w = re.sub(r"[^a-zA-Z?.!,Â¿]+", " ", w)
            w = w.strip()

            # Add sent to the list
            sentList.append(w)

    return sentList


# Get train_english and train_vietnamese
train_english = getSentList(train_english_path, 133260)
train_vietnamese = getSentList(train_vietnamese_path, 133260)

i = 0
while(True):
    if len(train_english[i]) == 0 or len(train_vietnamese[i]) == 0:
        del train_english[i]
        del train_vietnamese[i]
    i += 1
    if i >= len(train_english):
        break

i = 0
while(True):
    if len(train_english[i]) >= 100 or len(train_vietnamese[i]) >= 100:
        del train_english[i]
        del train_vietnamese[i]
    i += 1
    if i >= len(train_english):
        break

nltk.download('punkt')


words_english = []
# Need to modify this later
for sent in train_english:
    for word in word_tokenize(sent):
        words_english.append(word)

words_vietnamese = []
# Need to modify this later
for sent in train_vietnamese:
    for word in word_tokenize(sent):
        words_vietnamese.append(word)


# Create word2int and int2word dictionary
word_counter = collections.Counter(words_english).most_common()

word2int_english = dict()
word2int_english["<pad>"] = 0
word2int_english["<unk>"] = 1
word2int_english["<s>"] = 2
word2int_english["</s>"] = 3
for word, _ in word_counter:
    word2int_english[word] = len(word2int_english)

int2word_english = dict(
    zip(word2int_english.values(), word2int_english.keys()))

# Save word2int and int2word into pickle file
with open(word2int_english_path, 'wb') as f:
    pickle.dump(word2int_english, f)

with open(int2word_english_path, 'wb') as f:
    pickle.dump(int2word_english, f)


# Create word2int and int2word dictionary
word_counter = collections.Counter(words_vietnamese).most_common()

word2int_vietnamese = dict()
word2int_vietnamese["<pad>"] = 0
word2int_vietnamese["<unk>"] = 1
word2int_vietnamese["<s>"] = 2
word2int_vietnamese["</s>"] = 3
for word, _ in word_counter:
    word2int_vietnamese[word] = len(word2int_vietnamese)

int2word_vietnamese = dict(
    zip(word2int_vietnamese.values(), word2int_vietnamese.keys()))

# Save word2int and int2word into pickle file
with open(word2int_vietnamese_path, 'wb') as f:
    pickle.dump(word2int_vietnamese, f)

with open(int2word_vietnamese_path, 'wb') as f:
    pickle.dump(int2word_vietnamese, f)


# Convert input data from text to int
def get_intSeq_english(data_list, max_length, padding=False):
    seq_list = list()
    for sent in data_list:
        # Get tokens in each sent
        words = word_tokenize(sent)

        # Use this for train_english
        if(padding):
            # Make all sent to have the same length as max_length
            if(len(words) < max_length):
                words = words + (max_length-len(words))*["<pad>"]
            else:
                words = words[:max_length]

        # Use this for train_vietnamese
        else:
            words = words[:(max_length-1)]

        # Convert word to its corresponding int value
        # If the word doesnt exist, use the value of "<unk>" by default
        int_seq = [word2int_english.get(
            word, word2int_english["<unk>"]) for word in words]

        # Add int_seq to seq_list
        seq_list.append(int_seq)

    return seq_list

# Convert input data from text to int


def get_intSeq_vietnamese(data_list, max_length, padding=False):
    seq_list = list()
    for sent in data_list:
        # Get tokens in each sent
        words = word_tokenize(sent)

        # Use this for train_english
        if(padding):
            # Make all sent to have the same length as max_length
            if(len(words) < max_length):
                words = words + (max_length-len(words))*["<pad>"]
            else:
                words = words[:max_length]

        # Use this for train_vietnamese
        else:
            words = words[:(max_length-1)]

        # Convert word to its corresponding int value
        # If the word doesnt exist, use the value of "<unk>" by default
        int_seq = [word2int_vietnamese.get(
            word, word2int_vietnamese["<unk>"]) for word in words]

        # Add int_seq to seq_list
        seq_list.append(int_seq)

    return seq_list


# Define the max length of english and vietnamese
english_max_len = 100
vietnamese_max_len = 100

# Get the sequence of int value
train_english_intSeq = get_intSeq_english(
    train_english, english_max_len, padding=True)
train_vietnamese_intSeq = get_intSeq_vietnamese(
    train_vietnamese, vietnamese_max_len)


# load model
word_embed_english_w2v = KeyedVectors.load_word2vec_format(
    args["word_emb_src"], binary=True, unicode_errors='ignore')
# Sort the int2word
int2word_sorted = sorted(int2word_english.items())

# Get the list of word embedding corresponding to int value in ascending order
word_emb_list = list()
embedding_size = len(word_embed_english_w2v['the'])
for int_val, word in int2word_sorted:
    # Add Glove embedding if it exists
    if(word in word_embed_english_w2v):
        word_emb_list.append(word_embed_english_w2v[word])

    # Otherwise, the value of word embedding is 0
    else:
        word_emb_list.append(np.zeros([embedding_size], dtype=np.float32))

# Assign random vector to <s>, </s> token
word_emb_list[2] = np.random.normal(0, 1, embedding_size)
word_emb_list[3] = np.random.normal(0, 1, embedding_size)

# the final word embedding
word_embed_english = np.array(word_emb_list)


# load model
word_embed_vietnamese_w2v = KeyedVectors.load_word2vec_format(
    args["word_emb_targ"], binary=True, unicode_errors='ignore')

# Sort the int2word
int2word_sorted = sorted(int2word_vietnamese.items())

# Get the list of word embedding corresponding to int value in ascending order
word_emb_list = list()
embedding_size = len(word_embed_vietnamese_w2v['the'])
for int_val, word in int2word_sorted:
    # Add Glove embedding if it exists
    if(word in word_embed_vietnamese_w2v):
        word_emb_list.append(word_embed_vietnamese_w2v[word])

    # Otherwise, the value of word embedding is 0
    else:
        word_emb_list.append(np.zeros([embedding_size], dtype=np.float32))

# Assign random vector to <s>, </s> token
word_emb_list[2] = np.random.normal(0, 1, embedding_size)
word_emb_list[3] = np.random.normal(0, 1, embedding_size)

# the final word embedding
word_embed_vietnamese = np.array(word_emb_list)


def get_batches(input_data, output_data, batch_size):
    # Convert input and output data from list to numpy array
    input_data = np.array(input_data)
    output_data = np.array(output_data)

    # Number of batches per epoch
    num_batches_epoch = math.ceil(len(input_data)/batch_size)
    for batch_num in range(num_batches_epoch):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, len(input_data))
        yield input_data[start_index:end_index], output_data[start_index:end_index]


# Model for Machine Translation
class Seq2SeqModel(object):
    def __init__(self, vocab_size_en, vocab_size_vi, word_embedding_en, word_embedding_vi, input_len, output_len, params, train=True):
        # Get the vocab size
        self.vocab_size_en = vocab_size_en
        self.vocab_size_vi = vocab_size_vi

        # Get hyper-parameters from params
        self.num_layers = params['num_layers']
        self.num_hiddens = params['num_hiddens']
        self.learning_rate = params['learning_rate']
        self.keep_prob = params['keep_prob']
        self.beam_width = params['beam_width']

        # Using BasicLSTMCell as a cell unit
        self.cell = tf.nn.rnn_cell.LSTMCell

        # Define Place holders for the model
        self.batch_size = tf.placeholder(tf.int32, (), name="batch_size")
        # False means not adding the variable to the graph collection
        self.global_step = tf.Variable(0, trainable=False)

        # place holders for encoder
        self.inputSeq = tf.placeholder(tf.int32, [None, input_len])
        # Need to define the Shape as required in tf.contrib.seq2seq.tile_batch
        self.inputSeq_len = tf.placeholder(tf.int32, [None])

        # place holders for decoder
        self.decoder_input = tf.placeholder(tf.int32, [None, output_len])
        self.decoder_len = tf.placeholder(tf.int32, [None])
        self.decoder_target = tf.placeholder(tf.int32, [None, output_len])

        # Define projection_layer
        self.projection_layer = tf.layers.Dense(
            self.vocab_size_vi, use_bias=False)

        # Define the Embedding layer
        with tf.name_scope("embedding"):
            self.embeddings_en = tf.get_variable(
                "embeddings_en", initializer=tf.constant(word_embedding_en, dtype=tf.float32))
            self.embeddings_vi = tf.get_variable(
                "embeddings_vi", initializer=tf.constant(word_embedding_vi, dtype=tf.float32))

            # map the int value with its embeddings
            input_emb = tf.nn.embedding_lookup(
                self.embeddings_en, self.inputSeq)
            decoder_input_emb = tf.nn.embedding_lookup(
                self.embeddings_vi, self.decoder_input)

            # Convert from batch_size*seq_len*embedding to seq_len*batch_size*embedding to feed data with timestep
            # But, we need to set time_major=True during Training
            self.encoder_inputEmb = tf.transpose(input_emb, perm=[1, 0, 2])
            self.decoder_inputEmb = tf.transpose(
                decoder_input_emb, perm=[1, 0, 2])

        # Define the Encoder
        with tf.name_scope("encoder"):
            # Create RNN Cell for forward and backward direction
            fw_cells = list()
            bw_cells = list()
            for i in range(self.num_layers):
                fw_cell = self.cell(self.num_hiddens)
                bw_cell = self.cell(self.num_hiddens)

                # Add Dropout
                fw_cell = rnn.DropoutWrapper(
                    fw_cell, output_keep_prob=self.keep_prob)
                bw_cell = rnn.DropoutWrapper(
                    bw_cell, output_keep_prob=self.keep_prob)

                # Add cell to the list
                fw_cells.append(fw_cell)
                bw_cells.append(bw_cell)

            # Build a multi bi-directional model from fw_cells and bw_cells
            outputs, encoder_state_fw, encoder_state_bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                cells_fw=fw_cells, cells_bw=bw_cells, inputs=self.encoder_inputEmb, time_major=True, sequence_length=self.inputSeq_len, dtype=tf.float32)

            # The ouput of Encoder (time major)
            self.encoder_outputs = outputs

            # Use the final state of the last layer as encoder_final_state
            encoder_state_c = tf.concat(
                (encoder_state_fw[-1].c, encoder_state_bw[-1].c), 1)
            encoder_state_h = tf.concat(
                (encoder_state_fw[-1].h, encoder_state_bw[-1].h), 1)
            self.encoder_final_state = rnn.LSTMStateTuple(
                c=encoder_state_c, h=encoder_state_h)

        # Define the Decoder for training
        with tf.name_scope("decoder"):
            # Define Decoder cell
            decoder_num_hiddens = self.num_hiddens * 2  # As we use bi-directional RNN
            decoder_cell = self.cell(decoder_num_hiddens)

            # Training mode
            if(train):
                # Convert from time major to batch major
                attention_states = tf.transpose(
                    self.encoder_outputs, [1, 0, 2])

                # Decoder with attention
                attention = tf.contrib.seq2seq.BahdanauAttention(
                    num_units=decoder_num_hiddens, memory=attention_states, memory_sequence_length=self.inputSeq_len, normalize=True)
                attention_decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
                    cell=decoder_cell, attention_mechanism=attention, attention_layer_size=decoder_num_hiddens)

                # Use the final state of encoder as the initial state of the decoder
                decoder_initial_state = attention_decoder_cell.zero_state(
                    dtype=tf.float32, batch_size=self.batch_size)
                decoder_initial_state = decoder_initial_state.clone(
                    cell_state=self.encoder_final_state)

                # Use TrainingHelper to train the Model
                training_helper = tf.contrib.seq2seq.TrainingHelper(
                    inputs=self.decoder_inputEmb, sequence_length=self.decoder_len, time_major=True)
                decoder = tf.contrib.seq2seq.BasicDecoder(
                    cell=attention_decoder_cell, helper=training_helper, initial_state=decoder_initial_state, output_layer=self.projection_layer)
                logits, _, _ = tf.contrib.seq2seq.dynamic_decode(
                    decoder, output_time_major=True, maximum_iterations=output_len)

                # Convert from time major to batch major
                self.training_logits = tf.transpose(
                    logits.rnn_output, perm=[1, 0, 2])

                # Adding zero to make sure training_logits has shape: [batch_size, sequence_length, num_decoder_symbols]
                self.training_logits = tf.concat([self.training_logits, tf.zeros(
                    [self.batch_size, output_len - tf.shape(self.training_logits)[1], self.vocab_size_vi])], axis=1)

            # Inference mode
            else:
                # Using Beam search
                tiled_encoder_outputs = tf.contrib.seq2seq.tile_batch(tf.transpose(
                    self.encoder_outputs, perm=[1, 0, 2]), multiplier=self.beam_width)
                tiled_encoder_final_state = tf.contrib.seq2seq.tile_batch(
                    self.encoder_final_state, multiplier=self.beam_width)
                tiled_inputSeq_len = tf.contrib.seq2seq.tile_batch(
                    self.inputSeq_len, multiplier=self.beam_width)

                # Decoder with attention with Beam search
                attention = tf.contrib.seq2seq.BahdanauAttention(
                    num_units=decoder_num_hiddens, memory=tiled_encoder_outputs, memory_sequence_length=tiled_inputSeq_len, normalize=True)
                attention_decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
                    cell=decoder_cell, attention_mechanism=attention, attention_layer_size=decoder_num_hiddens)

                # Use the final state of encoder as the initial state of the decoder
                decoder_initial_state = attention_decoder_cell.zero_state(
                    dtype=tf.float32, batch_size=self.batch_size * self.beam_width)
                decoder_initial_state = decoder_initial_state.clone(
                    cell_state=tiled_encoder_final_state)

                # Build a Decoder with Beam Search
                beamSearch_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                    cell=attention_decoder_cell,
                    embedding=self.embeddings_vi,
                    start_tokens=tf.fill([self.batch_size], tf.constant(2)),
                    end_token=tf.constant(3),
                    initial_state=decoder_initial_state,
                    beam_width=self.beam_width,
                    output_layer=self.projection_layer
                )

                # Perform dynamic decoding with beamSearch_decoder
                outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
                    decoder=beamSearch_decoder, maximum_iterations=output_len, output_time_major=True)

                # Convert from seq_len*batch_size*beam_width to batch_size*beam_width*seq_len
                outputs = tf.transpose(outputs.predicted_ids, perm=[1, 2, 0])

                # Take the first beam (best result) as Decoder ouput
                self.decoder_outputs = outputs[:, 0, :]

        with tf.name_scope("optimization"):
            # Used for Training mode only
            if(train):
                # Caculate loss value
                masks = tf.sequence_mask(
                    lengths=self.decoder_len, maxlen=output_len, dtype=tf.float32)
                self.loss = tf.contrib.seq2seq.sequence_loss(
                    logits=self.training_logits, targets=self.decoder_target, weights=masks)

                # Using AdamOptimizer
                optimizer = tf.train.AdamOptimizer(self.learning_rate)
                # Compute gradient
                gradients = optimizer.compute_gradients(self.loss)
                # Apply Gradient Clipping
                gradients_clipping = [(tf.clip_by_value(
                    grad, clip_value_min=-1., clip_value_max=1.), var) for grad, var in gradients if grad is not None]

                # Apply gradients to variables
                self.train_update = optimizer.apply_gradients(
                    gradients_clipping, global_step=self.global_step)


# Define hyper-parameters for the Model
params = dict()
params['num_layers'] = int(args["num_layers"])
params['num_hiddens'] = int(args["num_hiddens"])
params['learning_rate'] = float(args["learning_rate"])
params['keep_prob'] = float(args["keep_prob"])
params['beam_width'] = int(args["beam_width"])

num_epochs = 10
early_stop = 5  # Stop if there is no improvement after 5 epochese
BATCH_SIZE = int(args["batch_size"])

# Set paths to save the model
checkpoint = args["checkpoint"]

start_time = time.time()

tf.reset_default_graph()

with tf.Session() as sess:
    # Create a Seq2seq model
    model = Seq2SeqModel(len(int2word_english), len(int2word_vietnamese), word_embed_english,
                         word_embed_vietnamese, english_max_len, vietnamese_max_len, params)

    # Initialize all variables
    sess.run(tf.global_variables_initializer())
    min_loss = 1000  # To find the minimum loss during training
    no_impove_count = 0  # Count the number of consecutive epoch having no improvement

    # load checkpoint
    saver = tf.train.Saver(tf.global_variables())
    saver.restore(sess, checkpoint)

    for epoch in range(num_epochs):
        # Get batches from training data
        batches = get_batches(train_english_intSeq,
                              train_vietnamese_intSeq, batch_size=BATCH_SIZE)

        # Reset epoch_loss after each epoch
        epoch_loss = 0
        # Interate over batches
        for batch_i, (batch_x, batch_y) in enumerate(batches):
            # The actual length of each sequence in the batch (excluding "<pad>")
            batch_x_len = list(
                map(lambda seq: len([word_int for word_int in seq if word_int != 0]), batch_x))

            # Decoder input is created by adding <s> to the begining of each output sentence
            batch_decoder_input = list(
                map(lambda seq: [word2int_vietnamese["<s>"]] + list(seq), batch_y))

            # The actual length of each Decoder input (excluding "<pad>")
            batch_decoder_len = list(map(lambda seq: len(
                [word_int for word_int in seq if word_int != 0]), batch_decoder_input))

            # The actual ouput of Decoder is created by adding </s> to the begining of each output sentence
            batch_decoder_output = list(map(lambda seq: list(
                seq) + [word2int_vietnamese["</s>"]], batch_y))

            # Add <pad> to make all input and ouput of Decoder have same length
            batch_decoder_input = list(
                map(lambda seq: seq + (vietnamese_max_len - len(seq)) * [word2int_vietnamese["<pad>"]], batch_decoder_input))
            batch_decoder_output = list(
                map(lambda seq: seq + (vietnamese_max_len - len(seq)) * [word2int_vietnamese["<pad>"]], batch_decoder_output))

            # Create a train_feed_dict
            train_feed_dict = {
                model.batch_size: len(batch_x),
                model.inputSeq: batch_x,
                model.inputSeq_len: batch_x_len,

                model.decoder_input: batch_decoder_input,
                model.decoder_len: batch_decoder_len,
                model.decoder_target: batch_decoder_output
            }

            # Start training the model
            _, step, loss, encoder_outputs = sess.run(
                [model.train_update, model.global_step, model.loss, model.encoder_outputs], feed_dict=train_feed_dict)
            epoch_loss += loss

            # Display loss value of each step
            print("step {0}: loss = {1}".format(step, loss))

        print("Finish epoch", epoch+1)
        # Averaging the epoch_loss
        epoch_loss = epoch_loss/(batch_i+1)

        # Save the model if the epoch_loss is at a new minimum,
        if epoch_loss <= min_loss:
            # Set new minimum loss
            min_loss = epoch_loss
            # Reset the no_impove_count
            no_impove_count = 0

            # Save the new model
            saver = tf.train.Saver(tf.global_variables())
            saver.save(sess, checkpoint)

            print('New model saved, minimum loss:', min_loss, '\n')

        # Early stopping
        else:
            print("No Improvement!", '\n')
            no_impove_count += 1
            if(no_impove_count == early_stop):
                print("Early stopping... Finish training")
                break

end_time = time.time()
training_time = (end_time-start_time)/60
print("\nTraining time (mins): ", training_time)
