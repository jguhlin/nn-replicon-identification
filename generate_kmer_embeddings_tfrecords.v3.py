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
import re

# k-mer size to use

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
k = 9
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
    return(np.array(data))
        
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

# print([len(all_kmers), len(unused_kmers), space])

# sys.exit(0)

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
        

# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.

filename = sys.argv[1]
print(filename)
strain_match = re.match("kmer-files/(.*)(.final.fasta|_filtered.fasta|.ASM584v2.dna.chromosome.Chromosome.fa)", filename)
strain = strain_match[1]
exit

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

writer = tf.python_io.TFRecordWriter("./kmer-embeddings/" + strain + ".tfrecord",
                                     options=tf.python_io.TFRecordOptions(2))

def embedding_kmer_generator(file, window_size):
    window_range = list(range(-window_size, 0))
    window_range.extend(list(range(1, window_size + 1)))
    
    def gen_training_data(idata, window_size):
        print(str(len(idata)))
        print(type(idata))
        startTime = time.time()
        for z in xrange(0, len(idata)):
            print(str(len(idata[z])))
            data = np.array(idata[z])
            sys.stdout.flush()
            for i in xrange(window_size, data.size - window_size):
                ikmer = kmer_dict[data[i]]
                for x in window_range:
                    if (x == (i + x)):
                        continue
                    kmer_pair = tf.train.Example(features=tf.train.Features(
                            feature={'x': _int64_feature(ikmer),
                                     'y': _int64_feature(kmer_dict[data[i + x]])}))
                    writer.write(kmer_pair.SerializeToString())
        elapsedTime = time.time() - startTime
        print(str(elapsedTime))
        
    gen_training_data(load_fasta(file), window_size)
    print(strain + " Calculated")



# print(len(kmer_dict))

# print("Writing kmers to TFRecords file")

window_size = 2

embedding_kmer_generator(filename, window_size)
        
# cProfile.run('embedding_kmer_generator("./rapid-start/", 4)')
