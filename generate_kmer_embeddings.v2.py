import tensorflow as tf
import numpy as np
import random
from six.moves import xrange  # pylint: disable=redefined-builtin
from Bio import SeqIO
import os
import concurrent.futures
import functools
import math
import pickle
import sys

# Version 2, with tensorboard support

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

# All kmer's in the dataset
# Already calculated so not recalculated
all_kmers = pickle.load(open("all_kmers.p", "rb"))
all_kmers = set(all_kmers)

kmer_dict = dict()
reverse_kmer_dict = dict();

a = 0
for i in all_kmers:
    kmer_dict[i] = a
    reverse_kmer_dict[a] = i
    a += 1


# Diff name to not be confused with replicon identification
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

def _add_layer_summary(value, tag):
  tf.summary.scalar('%s/fraction_of_zero_values' % tag, tf.nn.zero_fraction(value))
  tf.summary.histogram('%s/activation' % tag, value)    

batch_size = 8196
num_sampled = 1024

embedding_size = 1024
    
def main(unused_argv):
    classifier = tf.estimator.Estimator(
            model_fn=cnn_model_fn,
            model_dir="classifier_cnn_firsttry",
            config=tf.contrib.learn.RunConfig(
                    save_checkpoints_steps=5000,
                    save_checkpoints_secs=None,
                    save_summary_steps=1000))
    
#    tensors_to_log = {"probabilities": "softmax_tensor"}
#    logging_hook = tf.train.LoggingTensorHook(
#            tensors=tensors_to_log, every_n_iter=50)
        
    
    classifier.train(input_fn=my_input_fn)
#                     steps=10000
                     #hooks=[logging_hook])
    
    eval_results = classifier.evaluate(input_fn=my_input_fn, steps=1000)
    print(eval_results)
    
#  final_embeddings = normalized_embeddings.eval()
#  saver.save(session, './kmer-model', global_step=step)

    
if __name__ == "__main__":
  tf.app.run()

  
  np.save("final_embeddings" ,final_embeddings)
