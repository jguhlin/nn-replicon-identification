import tensorflow as tf
import numpy as np
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
import sys

# k-mer size to use
k = 9

# Guidance: the total genome size is size * 2 (reverse complement)
# Should make sure each iteration hits that total size, and preferably a few times
# It's ok to keep re-training the model
# Each iteration does batch_size * k...... 

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
   
# tf.logging.set_verbosity(tf.logging.INFO)

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
    data = list()
    file_base_name = os.path.basename(filename)
    picklefilename = file_base_name + ".kmerdict.picklepickle"
    if os.path.isfile(picklefilename):
        print("Loading from pickle")
        data = pickle.load(open(picklefilename, "rb"))
    else:
        print("File not found, generating new sequence: " + picklefilename)
        for seq_record in SeqIO.parse(filename, "fasta"):
            data.extend(get_kmers_from_seq(seq_record.seq.upper()))
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

# Because this was run at work on a smaller sample of files....
# with open("all_kmers_subset.txt", "w") as f:
#     for s in all_kmers:
#         f.write(str(s) +"\n")

# Because this was run at work on a smaller sample of files....
all_kmers = list()
# with open("all_kmers_subset.txt", "r") as f:
#     for line in f:
#         all_kmers.append(int(line.strip()))

all_kmers = pickle.load(open("all_kmers.p", "rb"))

all_kmers = set(all_kmers)
len(all_kmers)

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
    
kmer_count = len(all_kmers)

[len(all_kmers), len(unused_kmers), space]

def gen_random_training_data(input_data, window_size):
    idx = random.randrange(0, len(input_data))
    training_data = list();
    for i in range(idx - window_size, idx + window_size):
        if (i < 0): continue
        if (i >= len(input_data)): break
        if (i == idx): continue
        if type(input_data[idx]) == list: break;
        if type(input_data[i]) == list: break
        training_data.append([kmer_dict[input_data[idx]], kmer_dict[input_data[i]]])
    return training_data

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

# The current state is, each training batch is from a single FASTA file (strain, usually)
# This can be ok, as long as training batch is a large number
# Need to speed up reading of FASTA files though, maybe pyfaidx or something?

def generate_training_batch(data, batch_size, window_size):
#    data = list()
#    data = load_fasta(filefn())
    training_data = list();
    while len(training_data) < batch_size:
         training_data.extend(gen_random_training_data(random.choice(data), window_size))
    return training_data[:batch_size]
        

filegen = training_file_generator("data-files/")

training_data = load_fasta(filegen())

batch_size = 32768
embedding_size = 128
window_size = 4

validation_set = generate_training_batch(training_data, 100000, window_size)
validation_kmers = list(set([i[0] for i in validation_set]))
del validation_set

# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
valid_size = 2048
valid_examples = [validation_kmers[i] for i in np.random.choice(len(validation_kmers), valid_size, replace=False)]
del validation_kmers
num_sampled = 256

def embedding_kmer_generator(directory, window_size):
    files = [directory + "/" + f for f in os.listdir(directory)]
    random.shuffle(files)
    
    window_range = list(range(-window_size, 0))
    window_range.extend(list(range(1, window_size + 1)))

    def gen_training_data(idata, window_size):
        for data in idata:
            for i in xrange(window_size, len(data) + 1 - window_size):
                for x in window_range:
                    if (x == (i + x)):
                        continue
                    yield kmer_dict[data[i]], kmer_dict[data[i + x]]
    
    for f in files:
        yield from gen_training_data(load_fasta(f), window_size)

window_size = 4
def my_input_fn():
    kmer_gen = functools.partial(embedding_kmer_generator, "training-files/", window_size)

    ds = tf.data.Dataset.from_generator(kmer_gen, 
                                        (tf.int64,
                                         tf.int64),
                                        (tf.TensorShape(1),
                                         tf.TensorShape(None)))
                                        
    # Numbers reduced to run on my desktop
    #ds = ds.repeat(4)
    #ds = ds.prefetch(5000000)
    #ds = ds.shuffle(buffer_size=500000)
    #ds = ds.batch(8000)
    
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

graph = tf.Graph()

with graph.as_default():

  # Input data.
  train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
  train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
  valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
  
  # Ops and variables pinned to the CPU because of missing GPU implementation
  # Look up embeddings for inputs.
  embeddings = tf.Variable(
      tf.random_uniform([kmer_count, embedding_size], -1.0, 1.0))
  embed = tf.nn.embedding_lookup(embeddings, train_inputs)

  # Construct the variables for the NCE loss
  nce_weights = tf.Variable(
      tf.truncated_normal([kmer_count, embedding_size],
                          stddev=1.0 / math.sqrt(embedding_size)))
  nce_biases = tf.Variable(tf.zeros([kmer_count]))

# Compute the average NCE loss for the batch.
  # tf.nce_loss automatically draws a new sample of the negative labels each
  # time we evaluate the loss.
  loss = tf.reduce_mean(
      tf.nn.nce_loss(weights=nce_weights,
                     biases=nce_biases,
                     labels=train_labels,
                     inputs=embed,
                     num_sampled=num_sampled,
                     num_classes=kmer_count))

  # Construct the SGD optimizer using a learning rate of 1.0.
  optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)
  # optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)

  # Compute the cosine similarity between minibatch examples and all embeddings.
  norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
  normalized_embeddings = embeddings / norm
  valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
  similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

  # Add variable initializer.
  init = tf.global_variables_initializer()
  saver = tf.train.Saver()

num_steps = 150000

print("Loading initial batch data, this could take a few minutes")

executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
future = executor.submit(load_fasta, filegen())

with tf.Session(graph=graph, config=tf.ConfigProto(log_device_placement=False)) as session:
  # We must initialize all variables before we use them.
  init.run()
  writer = tf.train.SummaryWriter("./kmer-model", session.graph)
  print('Initialized')

  average_loss = 0
  for step in xrange(num_steps):
    
    if step % 500 == 0: # Change files every 0.5k steps # Batch size is large enough right now....
        print("Loading new file at step: ", step)
        # Start loading the next file, so it has time to finish while the neural net does its training
        tdata = future.result()
        future = executor.submit(load_fasta, filegen())
        sys.stdout.flush()
        
    if len(tdata) == 0:
        print("Using short-circuit load-fasta at step: ", step)
        tdata = load_fasta(filegen()) # Emergency short-circuit here....
        
    batch_data = generate_training_batch(tdata, batch_size, window_size)
    feed_dict = {train_inputs: [x[0] for x in batch_data], 
                 train_labels: [[x[1]] for x in batch_data]}

    # We perform one update step by evaluating the optimizer op (including it
    # in the list of returned values for session.run()
    _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
    average_loss += loss_val

    # Print status every 1k steps
    if step % 1000 == 0:
        if step > 0:
            average_loss /= 1000
            # The average loss is an estimate of the loss over the last 2000 batches.
        print('Average loss at step ', step, ': ', average_loss)
        tf.summary.merge_all()
        
        average_loss = 0
        sys.stdout.flush()
    
    # Save every 20k steps
    if step % 500 == 0:
        print("Saving model at step: ", step)
        tf.summary.merge_all()
        saver.save(session, './kmer-model', global_step=step)
        print("Saved model at step: ", step)
        sys.stdout.flush()

        
#    if step % 20000 == 0:
#        sim = similarity.eval()
#        accuracy = 0
#        for i in range(0, 100):
#            rand_kmer = random.choice(list(validation_dict.keys()))
#            top_k = 10
#            nearest = (-sim[rand_kmer, :]).argsort()[1:top_k + 1]
            

  final_embeddings = normalized_embeddings.eval()
  tf.summary.merge_all()
  saver.save(session, './kmer-model', global_step=step)
  
  np.save("final_embeddings" ,final_embeddings)
