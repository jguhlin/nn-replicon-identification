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
from collections import deque
import itertools
import re


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

    for key in data:
        data[key] = np.array(data[key])
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

# Because this was run at work on a smaller sample of files....
# with open("all_kmers_subset.txt", "w") as f:
#     for s in all_kmers:
#         f.write(str(s) +"\n")

sess = tf.Session()

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
                yield(kentry, [repdict[k]])

# Need this for the following 2 fns....
replicons_list = get_categories("training-files/")
repdict = dict()
a = 0
for i in replicons_list:
    repdict[i] = a
    a += 1

# Not infinite
def kmer_generator(directory, window_size):
    files = [directory + "/" + f for f in os.listdir(directory)]
    random.shuffle(files)
    for f in files:
        fasta = load_fasta(f)
        yield from gen_training_data_generator(fasta, window_size, repdict)
        
def kmer_generator_file(file):
    fasta = load_fasta(file)
    yield from gen_training_data_generator(fasta, window_size, repdict)

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

files = ["training-files/" + f for f in os.listdir("training-files/")]


for f in files:
    strain_match = re.match("training-files/(.*)(.final.fasta|_filtered.fasta|.ASM584v2.dna.chromosome.Chromosome.fa)", f)
    strain = strain_match[1]
    
    writer = tf.python_io.TFRecordWriter("./training-tfrecords/" + strain + ".tfrecord",
                                     options=tf.python_io.TFRecordOptions(2))

    for x,y in kmer_generator_file(f):
        kmers = tf.train.Example(features=tf.train.Features(
                feature={'x': _int64_feature(x),
                         'y': _int64_feature(y)}))
        writer.write(kmers.SerializeToString())