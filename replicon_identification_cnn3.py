# Next step is to add filename processed to text summary

import tensorflow as tf
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from collections import Counter
import collections
import random
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import Bio
from Bio import SeqIO
import os
import concurrent.futures
import functools
from functools import partial
import math
import threading
import time
import random
from random import shuffle
import pickle
import ntpath
import os.path
import sys
from tensorflow.python.summary import summary


# k-mer size to use
k = 9

#
# NOTE!!!!!!!!!!!!!!!!
#
# We can reduce problem space if we get the reverse complement, and add a bit to indicate reversed or not...
# Not really.... revcomp just doubles it back up again....
#
# Also -- Build a recurrent network to predict sequences that come after a given kmer?
# Look at word2vec, dna2vec, bag of words, skip-gram
#

# Problem space
space = 5 ** k

def partition(n, step, coll):
    for i in range(0, len(coll), step):
        if (i+n > len(coll)):
            break #  raise StopIteration...
        yield coll[i:i+n]
        
def get_kmers(k):
    return lambda sequence: partition(k, k, sequence)

def convert_nt(c):
    return {"N": 0, "A": 1, "C": 2, "T": 3, "G": 4}.get(c, 0)

def convert_nt_complement(c):
    return {"N": 0, "A": 3, "C": 4, "T": 1, "G": 2}.get(c, 0)

def convert_kmer_to_int(kmer):
    return int(''.join(str(x) for x in (map(convert_nt, kmer))), 5)

def convert_kmer_to_int_complement(kmer):
    return int(''.join(str(x) for x in reversed(list(map(convert_nt_complement, kmer)))), 5)

def convert_base5(n):
    return {"0": "N", "1": "A", "2": "C", "3": "T", "4": "G"}.get(n,"N")

def convert_to_kmer(kmer):
    return ''.join(map(convert_base5, str(np.base_repr(kmer, 5))))

# Not using sparse tensors anymore.
   
tf.logging.set_verbosity(tf.logging.INFO)

# Get all kmers, in order, with a sliding window of k (but sliding 1bp for each iteration up to k)
# Also get RC for all....

def kmer_processor(seq,offset):
    return list(map(convert_kmer_to_int, get_kmers(k)(seq[offset:])))

def get_kmers_from_seq(sequence):
    kmers_from_seq = list()

    kp = functools.partial(kmer_processor, sequence)
    
    for i in map(kp, range(0,k)):
        kmers_from_seq.append(i)

    rev = sequence[::-1]
    kpr = functools.partial(kmer_processor, rev)
    
    for i in map(kpr, range(0,k)):
        kmers_from_seq.append(i)
            
#    for i in range(0,k):
#        kmers_from_seq.append(kmer_processor(sequence,i))
#    for i in range(0,k):
#        kmers_from_seq.append(kmer_processor(rev, i))
    return kmers_from_seq

data = list()

def load_fasta(filename):
    # tf.summary.text("File", tf.as_string(filename))
    data = dict()
    file_base_name = ntpath.basename(filename)
    picklefilename = file_base_name + ".picklepickle"
    if os.path.isfile(picklefilename):
        print("Loading from pickle: " + filename)
        data = pickle.load(open(picklefilename, "rb"))
    else:
        print("File not found, generating new sequence: " + picklefilename)
        for seq_record in SeqIO.parse(filename, "fasta"):
            data.update({seq_record.id:
                         get_kmers_from_seq(seq_record.seq.upper())})
        pickle.dump(data, open(picklefilename, "wb"))
    sys.stdout.flush()
    return(data)
        
def get_kmers_from_file(filename):
    kmer_list = list()
    for seq_record in SeqIO.parse(filename, "fasta"):
        kmer_list.extend(get_kmers_from_seq(seq_record.seq.upper()))
    return set([item for sublist in kmer_list for item in sublist])

all_kmers = set()

# Very slow, should make this part concurrent...

def find_all_kmers(directory):
    kmer_master_list = list()
    files = [directory + "/" + f for f in os.listdir(directory)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        for i in executor.map(get_kmers_from_file, files):
            kmer_master_list.extend(list(i))
            kmer_master_list = list(set(kmer_master_list))
            print("Total unique kmers: " + str(len(set(kmer_master_list))))
    return set(kmer_master_list)

def get_categories(directory):
    data = list()
    files = os.listdir(directory)
    for filename in files:
        for seq_record in SeqIO.parse(directory + "/" + filename, "fasta"):
            data.append(seq_record.id)
    data = sorted(list(set(data)))
    return(data)

def training_file_generator(directory):
    files = [directory + "/" + f for f in os.listdir(directory)]
    random.shuffle(files)
    def gen():
        nonlocal files
        if (len(files) == 0):
            files = [directory + "/" + f for f in os.listdir(directory)]
            random.shuffle(files)
        return(files.pop())
    return gen

def gen_random_training_data(input_data, window_size):
    rname = random.choice(list(input_data.keys()))
    rdata = random.choice(input_data[rname])
    idx = random.randrange(window_size + 1, len(rdata) - window_size - 1)
    tdata = list();
    for i in range(idx - window_size - 1, idx + window_size):
        if (i < 0): continue
        if (i >= len(rdata)): break
        if type(rdata[idx]) == list: break;
        if type(rdata[i]) == list: break
        tdata.append(kmer_dict[rdata[i]])
    return tdata, rname

# The current state is, each training batch is from a single FASTA file (strain, usually)
# This can be ok, as long as training batch is a large number
# Need to speed up reading of FASTA files though, maybe pyfaidx or something?

# Define the one-hot dictionary...

replicons_list = get_categories("training-files/")

oh = dict()
a = 0
for i in replicons_list:
    oh[i] = tf.one_hot(a, len(replicons_list))
    a += 1
    
oh = dict()
a = 0
for i in replicons_list:
    oh[i] = a
    a += 1
    
oh = dict()
oh['Main'] = [1.0, 0.0, 0.0]
oh['pSymA'] = [0.0, 1.0, 0.0]
oh['pSymB'] = [0.0, 0.0, 1.0]


def generate_training_batch(data, batch_size, window_size):
    training_batch_data = list();
    while len(training_batch_data) < batch_size:
         training_batch_data.append(gen_random_training_data(data, 
                                                             window_size))
    return training_batch_data

batch_size = 1024
embedding_size = 128
window_size = 7

replicons_list = get_categories("training-files/")

filegen = training_file_generator("training-files/")

repdict = dict()
a = 0
for i in replicons_list:
    repdict[i] = a
    a += 1

def test_input_fn(data):
    tbatch = generate_training_batch(data, 1, window_size)
    dat = {'x': tf.convert_to_tensor([tf.convert_to_tensor(get_kmer_embeddings(tbatch[0][0]))])}
    lab = tf.convert_to_tensor([repdict[tbatch[0][1]]])
    return dat, lab

def train_input_fn_raw(data):
    tbatch = generate_training_batch(data, 1, window_size)
    dat = {'x': (get_kmer_embeddings(tbatch[0][0]))}
    lab = [repdict[tbatch[0][1]]]
    return dat, lab

sess = tf.Session()

all_kmers = list()
all_kmers = pickle.load(open("all_kmers.p", "rb"))

all_kmers = set(all_kmers)
len(all_kmers)
unused_kmers = set(range(0, space)) - all_kmers

kmer_dict = dict()
reverse_kmer_dict = dict();

a = 0
for i in all_kmers:
    kmer_dict[i] = a
    reverse_kmer_dict[a] = i
    a += 1
    
kmer_count = len(all_kmers)

[len(all_kmers), len(unused_kmers), space]

# This fn now generates all possible combinations of training data....

def gen_training_data(input_data, window_size):
    total_data = list()
    
    for k in input_data.keys():
        for kdata in input_data[k]:
            for i in range(window_size + 1, len(kdata) - window_size):
                kentry = list()
                for x in range(i - window_size - 1, i + window_size):
                    kentry.append(kmer_dict[kdata[x]])
                total_data.append([kentry, k])
    return total_data

feature_columns = [tf.feature_column.numeric_column("x", shape=[15,128])]

embeddings = np.load("final_embeddings.npy")

def get_kmer_embeddings(kmers):
    a = list() # numpy.empty(128 * 15)
    for k in kmers:
        a.append(embeddings[k])
    return a
    #return np.hstack(a)

def gen_training_data_generator(input_data, window_size, repdict):
    for k in input_data.keys():
        for kdata in input_data[k]:
            for i in range(window_size + 1, len(kdata) - window_size):
                kentry = list()
                for x in range(i - window_size - 1, i + window_size):
                    kentry.append(kmer_dict[kdata[x]])
                yield(get_kmer_embeddings(kentry), [repdict[k]])

# Not infinite
def kmer_generator(directory, window_size):
    files = [directory + "/" + f for f in os.listdir(directory)]
    random.shuffle(files)
    
    replicons_list = get_categories("training-files/")
    repdict = dict()
    a = 0
    for i in replicons_list:
        repdict[i] = a
        a += 1
    
    for f in files:
        yield from gen_training_data_generator(load_fasta(f), window_size, repdict)
        
# Plan to use tf.data.Dataset.from_generator
# ds = tf.contrib.data.Dataset.list_files("training-files/").map(tf_load_fasta)


def my_input_fn():
    kmer_gen = functools.partial(kmer_generator, "training-files/", window_size)

    ds = tf.data.Dataset.from_generator(kmer_gen, 
                                        (tf.float32,
                                         tf.int64),
                                        (tf.TensorShape([15,128]),
                                         tf.TensorShape(None)))
                                        
#    # Numbers reduced to run on my desktop
#    ds = ds.repeat(5)
#    ds = ds.prefetch(5000) # Each batch is only 2048, so prefetch 5000
#    ds = ds.shuffle(buffer_size=1000000) # Large buffer size for better randomization
#    ds = ds.batch(2048) # Reduced from 5000 so it runs quicker
    
    ds = ds.repeat(1)
    ds = ds.prefetch(1000)
    ds = ds.shuffle(buffer_size=500)
    ds = ds.batch(250)
    
    def add_labels(arr, lab):
        return({"x": arr}, lab)
    
    ds = ds.map(add_labels)
    iterator = ds.make_one_shot_iterator()
    batch_features, batch_labels = iterator.get_next()
    return batch_features, batch_labels
    
# CNN experiment for training
# Based off of kmer embeddings

def _add_layer_summary(value, tag):
  summary.scalar('%s/fraction_of_zero_values' % tag, tf.nn.zero_fraction(value))
  summary.histogram('%s/activation' % tag, value)    

# For third try we are changing the model to something based off of word embeddings,
# rather than something based off of image concepts and slightly altered (v1 -> v2)
  

# Based off of: https://www.tensorflow.org/tutorials/layers
def cnn_model_fn(features, labels, mode):
    """Model fn for CNN"""
    
    # Input layer
    # So inputs are 1920, or 15 * 128, and "1" deep (which is a float)
    input_layer = tf.reshape(features["x"], [-1, 15, 128, 1])
    
    # filters * kernelsize[0] * kernel_size[1] must be > input_layer_size
    # So 1920 <= 32 * 5 * 12
    # 32 dimensions, 5 x 12 sliding window over entire dataset
    conv1 = tf.layers.conv2d(
            inputs=input_layer,
            filters=32,
            kernel_size=[4,32],
            strides=5,
            padding="same",
            activation=tf.nn.relu)
    
    # Convolutional Layer #2 from input layer
    conv2 = tf.layers.conv2d(
            inputs=input_layer,
            filters=32,
            kernel_size=[6, 64],
            strides=5,
            padding="same",
            activation=tf.nn.relu)
    
    conv3 = tf.layers.conv2d(
            inputs=input_layer,
            filters=32,
            kernel_size=[8, 256],
            strides=5,
            padding="same",
            activation=tf.nn.relu)
    
    
    print(tf.shape(conv1))
    print(tf.shape(conv2))
    print(tf.shape(conv3))
    
    concatenated = tf.concat([conv1, conv2, conv3], 3)
    
    # 1,024 neurons
    dense = tf.layers.dense(inputs=concatenated, units=1024, activation=tf.nn.relu)

    _add_layer_summary(dense, "Dense")
    
    # Gonna try this but dropout is very high (was 0.4, now 0.2)
    dropout = tf.layers.dropout(
            inputs=dense, rate=0.2, training=mode == tf.estimator.ModeKeys.TRAIN)
    
    # Must have len(replicons_list) neurons
    logits = tf.layers.dense(inputs=dropout, units=len(replicons_list))
    
    _add_layer_summary(logits, "Logits")

    predictions = {
            "classes": tf.argmax(input=logits, axis=1),
             "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
             }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    
    # Calculate Loss (for both TRAIN and EVAL modes)
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=len(replicons_list))
    loss = tf.losses.softmax_cross_entropy(
            onehot_labels=onehot_labels, logits=logits)
    
    correct_prediction = tf.equal(tf.argmax(logits, 1), labels)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
 
    #f.summary.text("LogitsArgMax", tf.as_string(tf.argmax(logits, 1)))
    #tf.summary.text("Labels", tf.as_string(labels))
    #tf.summary.text("Prediction", tf.as_string(tf.argmax(labels, 1)))
    
    
#    tf.summary.text("Onehot", tf.as_string(onehot_labels))
#    tf.summary.text("Predictions", tf.as_string(correct_prediction))
    
    tf.summary.scalar('Accuracy', accuracy)

    
    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.002)
        train_op = optimizer.minimize(
                loss=loss,
                global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
   
    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
            "accuracy": tf.metrics.accuracy(
                    labels=labels, predictions=predictions["classes"])}
    
    return tf.estimator.EstimatorSpec(
            mode=mode, 
            loss=loss, 
            eval_metric_ops=eval_metric_ops, 
            summary_op=tf.summary.merge_all())
    
def main(unused_argv):
    classifier = tf.estimator.Estimator(
            model_fn=cnn_model_fn,
            model_dir="classifier_cnn3",
            config=tf.contrib.learn.RunConfig(
                    save_checkpoints_steps=1500,
                    save_checkpoints_secs=None,
                    save_summary_steps=100))
    
#    tensors_to_log = {"probabilities": "softmax_tensor"}
#    logging_hook = tf.train.LoggingTensorHook(
#            tensors=tensors_to_log, every_n_iter=50)
        
    
    classifier.train(input_fn=my_input_fn)
#                     steps=10000
                     #hooks=[logging_hook])
    
    eval_results = classifier.evaluate(input_fn=my_input_fn, steps=1000)
    print(eval_results)
    
if __name__ == "__main__":
  tf.app.run()





