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
import tempfile
import ntpath
import os.path

# IF you change the kmer size here, make sure to regenerate
# the kmer embeddings and change the k there!
# k-mer size to use
k = 9
# Problem space
space = 5 ** k

batch_size = 1024
embedding_size = 128
window_size = 7

# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
valid_size = 1024
num_sampled = 256

learning_rate = 0.1

# Network Parameters
n_hidden_1 = 500
n_hidden_2 = 50



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
    data = dict()
    file_base_name = ntpath.basename(filename)
    picklefilename = file_base_name + ".picklepickle"
    if os.path.isfile(picklefilename):
        print("Loading from pickle")
        data = pickle.load(open(picklefilename, "rb"))
    else:
        print("File not found, generating new sequence: " + picklefilename)
        for seq_record in SeqIO.parse(filename, "fasta"):
            data.update({seq_record.id:
                         get_kmers_from_seq(seq_record.seq.upper())})
        pickle.dump(data, open(picklefilename, "wb"))
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

# len(data)

# all_kmers = set([item for sublist in data for item in sublist])
unused_kmers = set(range(0, space)) - all_kmers

kmer_dict = dict()
reverse_kmer_dict = dict();

a = 0
for i in all_kmers:
    kmer_dict[i] = a
    reverse_kmer_dict[a] = i
    a += 1

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

def train_input_fn():
    rdata = generate_training_batch(training_data, 1, window_size)[0]
    return rdata[0], oh[rdata[1]]
    # return {"train_input": rdata[0]}, oh[rdata[1]]

filegen = training_file_generator("training-files/")
training_data = load_fasta(filegen())
validation_set = generate_training_batch(training_data, 10000, window_size)
# validation_kmers = list(set([i[0] for i in validation_set]))
# del validation_set
valid_examples = [i[0] for i in validation_set]
del validation_set

# Load embeddings
embeddings = np.load("embeddings_200000.npy")

# graph = tf.Graph()

def model_fn(features, labels, mode):
    kmers = tf.Variable(tf.constant(0.0, shape=[kmer_count, 128]),
                        trainable=False, name="kmers")
    
    embedding_placeholder = tf.placeholder(tf.int32, [kmer_count, 128])
    embedding_init = kmers.assign(embeddings)
    
    train_input = tf.placeholder(tf.int32, shape=[batch_size, 15]) 
    train_label = tf.placeholder(tf.int32, shape=[batch_size, 3])
    
    kmer_input = tf.nn.embedding_lookup(embeddings, train_input)
    kmer_input_r = tf.reshape(kmer_input, [batch_size, -1])
    
    l1 = tf.layers.dense(kmer_input_r, n_hidden_1)
    l2 = tf.layers.dense(l1, n_hidden_2)
    logits = tf.layers.dense(l2, len(replicons_list))
    
    pred_classes = tf.argmax(logits, axis=1)
    pred_prob = tf.nn.softmax(logits)
    
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = train_label))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    acc = tf.metrics.accuracy(labels = tf.argmax(train_label, 1), predictions=pred_classes)

    estim_specs = tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=pred_classes,
            loss=loss,
            train_op=train,
            eval_metric_ops={'accuracy': acc})

    return estim_specs

        
        
        
    