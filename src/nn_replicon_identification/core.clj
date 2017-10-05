(ns nn-replicon-identification.core
  (:require [clojure.core.matrix :as mat]
            [clojure.core.reducers :as r]
            [clj-fuzzy.metrics :as fm]
            [cortex.experiment.train :as train]
            [cortex.nn.layers :as layers]
            [cortex.nn.network :as network]
            [cortex.nn.execute :as execute]
            [cortex.optimize.adadelta :as adadelta]
            [cortex.optimize.adam :as adam]
            [cortex.metrics :as metrics]
            [cortex.util :as util]
            [iota :as iota]
;            [clj-btable.core :as btable]
            [cortex.optimize.adam :as adam]
            [cortex.experiment.train :as experiment-train]
            [cortex.nn.execute :as execute]
            [criterium.core :as crit]
            [clojure.math.numeric-tower :as math]
            [biotools.fasta :as fasta]))

(mat/set-current-implementation :vectorz)

(def k 7)
(def space (math/expt 5 k))

; Use base 5, 0 is \N, etc...
;(Integer/toString 1953124 5)
; Build sparse-array 

(defn get-kmers [k]
  (fn [sequence]
    (distinct
      (partition k 1 sequence)))) ; Could move sliding window by 1

(defn convert-char-to-number [c]
  (case c 
    \N 0 \A 1 \C 2 \T 3 \G 4 0))

(defn convert-kmer [kmer]
  (new BigInteger (apply str (map convert-char-to-number kmer)) 5))

(def seq1 "ACTGGGCTAATACCAATTAACAGATTGAGATTTAATGATGACAATAGA")
(def seq2 "TTTAAAAAGTAATACCAATTAACAGATTGAGATTTAATGATGACAATAGAACTGACAATACA")
(def seq-main
  (clojure.string/upper-case    "CTCTGGAAAGGAGATTCGGCAGTGCGAAGGCGAGGCCCGCAAGGACGACCAGCCAGATGAGAATATTCTTGAACGGCGACAACCTGGGCATATTTCGAAATGCTCCATCCGCCGGTCTCTGCCGGTCGCGGCCCTTGCCCTAACCACGGTGATTTTGGCCTCGCGTCGCTCCAAAATCAAACCGTGATCAATGCCGATCGAAGGCGCGGTATGCGGGCGAAAACCGCGCACGCTTCTTCAAGTACCCGCGCCGGGGTTACTCCTTGACGGGCTCGCCCTTCACACGAACCTCGGAGACGCCGCTGCGCACGACGCGGACCTTGATGCCCTCGGCGATCTCCACTTCGAGCTCCGTATCGTCGACGACCTTGGTCACCTTGCCGACAATGCCGCCGCCGGTGACGACCTGGTCGCCGCGCCGGATGTTCTTCAGGAGCTCTTCGCGGCGCTTCATCTGCGCACGCTGCGGCCGGATGATCAGGAAATACATCACCACGAAGATCAGCAGGAATGGCAGGATGGACATCAGAATGTCGGCGCCGCCGCCCCAGGGGCCGCCGTCTGCGCGAAAGCTTCGGTAATAAACATCGATCACTCCTTGAGTTCAAATTGCGCGCTTGCCCCCGCGGCAAACCTGCCGGAATATAGGCAAGCCGTCCCGTAACACAAATCGTCGGTACACTTCCCCGTTTCTCCTGCCTCTGGCACAAATTCCGCAGCAGGAGAACCCCCTGGTTGCAGGCTGCCGGTCTTTTCCAGCGCAAACCGCCGTGCTACCGAGAAAAACGCCGCCGGCGGCAGCTTCAACGGATTCGACCGGAGGATGAACGTCGCGGCGATTCAAGGATTTGGGTGGGCTGACGCACGCCCGTTATCGCGCGGGTCGGCCGCCCAGTTCGAAATTCAGCCTGCCGGAGATACATGAAATGCCCGAAAGCAAGATCGACGTCCTGCTCAACGAAATACAGAAGCTTTCGGCCGCGATGGAGCGCATCGCCGGACCGGCATATGCCGTCAACAATTGGCATGAGGCGGAGTGTTTCGTCTGGGCACCGGCCACGCGCCACCTGCAGCCCGTCCCGAGGCCGAATCGCATCGACCTCGCGCTCATCGCCGGCGTCGACCATGTCCGCGACATTCTCTTCGACAACACGCTCCGCTTCGCCGAAGGCTATCCGGCGAACAACGTGCTCCTGTGGGGCGCCCGCGGCATGGGCAAATCGTCACTGGTCAAGGCGGTCCACGCAAAGGTCGCCCACGACACCGGCAGCGCAATCAAGCTTGTCGAAGTACACCGGGAGGATATCGCCACCCTGCCCGTGCTGATGGAAATCCTGAAGGCGGCGCCGATGCCCGTGATCGTCTTCTGCGATGATCTCTCCTTCGATCACGACGATACCTCCTACAAGTCGCTGAAGGCGGTTCTCGACGGCGGCGTCGAGGGGCGCCCGGCAAACGTTCTGCTCTATGCGACGTCCAACCGCAGACACCTGCTTCCCCGCAACATGATGGAAAATGAACAATCCACCGCCATTAACCCCTCGGAGGCCGTCGAGGAAAAAGTGTCGCTATCCGACCGCTTCGGGCTATGGCTGGGTTTCTACAAGTGCAGCCAGGACGACTATCTGGCGATGGTCGACGGGTATGCGCAGTACTTCAAATTGCCTCTCGAGCCCGAAGCGCTGCATGCCGAGGCTCTTGAATGGGCGACGACGCGAGGATCGAGGTCCGGCCGCGTCG"))

(def seq-psyma
  (clojure.string/upper-case  "GAACACCGGTACGGCGCCGAGCGCATCGACCTTCGACAGCCTGCTCGACAAGGGACAGGCCTCAGCCACCGATATTTGGTCACGTGCCTCCTGGCCGGTCGACATCGTCACCGGCGTCGGCGGCATGATGGTGATCGGCGCGAGCTTCATCGTCGCCGCGATCGGCTATATCGTCTCGCTTTACGCGCGGCTGGCGCTTGCCATCGTGCTCGCGATTGGACCAATTTTCGTGGCGCTCGCCATGTTTCAGGCGACGCGGCGCTTCACGGAGGCATGGATCGGCCAGCTTGCGAACTTTGTGATCCTCCAGGTCCTCGTCGTCGCCGTCGGCTCTCTACTGATCACCTGCATCGACACCACCTTCGCGGCGATCGACGGATATAGCGATGTGCTGATGCGGCCGATCGCACTCTGCGCCATCTGCCTCGCGGCTCTCTATGTCTTCTATCAACTCCCGAACATCGCCTCGGCGCTTGCCGCCGGCGGCGCGTCGTTGACCTACGGCTACGGCGCCGCACGCGACGCCCACGAAAGCACGCTCGCCTGGGCGGCTTCCCATACCGTCCGTGCGGCCGGACGTGGTGTCCGTGCCGTTGGCCGAACCTTCACCTCAAAAGGCTCCGGATCATGACGCTTTTCGCACGAACAAGAGAAAGGCTTTCCAGGATTAATCAGAACGTTCCGCTGCTTTGCGTTGCGGCGATCTTAAGCGGTTGCGCATCGATGACCTATCCGCTCCCGAAATGTGACGGCTATTCGCGCCGGCCCCTCAATCGATCGATGTGGCAGTGGGAAGACAATAGCAACTTCAAGCTGAAACAGTCCGATGCGCGACCGGCGGCCTCTCAGTCCGTCGCCACCGCTTATGCCGGCGAGGGCAGGGAATTTCCCGCCTTCGCACATCTCGACATCGACGCATCCTATCGTCCTTGCGAGGGTTGACTCGATGGTCTCGGCGGACGAACTCAAGACATACTTCGAAAAGGCGCGACGCTTCGATCAGGACCGCGTGATCCAGGT"))

(def seq-psymb 
  (clojure.string/upper-case  "cgcCGCGGCTGCGGTTCAGCGCCAGCTCCAGATTGTCCCAGACCGTATGGTTCTCGAAGACGGTCGGCTTCTGGAACTTGCGGCCGATGCCGAGCTCGGCGATTGCCGCTTCGTCTTTCTTGGTGAGGTCGATGTCGCCCTTGAAGAAGACCTCGCCCTCGTCCGGCCGCGTCTTGCCGGTGATGATGTCCATCATCGTCGTCTTGCCGGCGCCATTGGGGCCGATGATCGCGCGCAGTTCCCCCGGCTCTACGACGAAGGAGAGCGAGTTTAGCGCCTTGAAGCCATCGAAGGAGACGGAGACCCCATCGAGATAGAGCAGGTTCCTGGGTTTCTTTCCGGTCATGGCGATCACTCCGCGGCCACCGTTTCGGCGTCCGCAAGGCTCGCCGCTTTTTCGCTCTCGCTTTCCTTCCGGGCCGCCGCGTGGGATGTGCGCCGGCTTGCGAGATAGCTCTGCGCCGTGCCGACCACGCCCTTCGGCAGGAAAAGCGTGACGAGGACGAAGAGCCCGCCGAGCGCAAAGAGCCAGAATTCGGGGAAGGCGGCGGTGAATATGCTTTTTCCGCCGTTGACGAGGATCGCGCCGACGATCGGTCCGATCAGCGTGCCGCGCCCGCCGACAGCCGTCCATATGACCACCTCGATCGAATTGGCGGGGGCGAACTCGCCCGGATTGATGATGCCGACTTGCGGCACGTAGAGCGCGCCGGCGACGCCCGCCATCATTGCCGAGACCGTGAAGGCGAAGAGCTTCATGTGCTCGACGCGATAGCCGAGAAAGCGTGTGCGGCTTTCCGCGTCGCGCAGCGCCACCAGCACCTTGCCGAATTTCGAGCGGACGATGCCCGAGGTGACGACGAGCGAAACGGCAAGCGCCAGCGCGGAGGCTGCAAAGAGTGCCGCACGCGTTCCGTCGGCCTGGATGTTGAAGCCGAGGATGTCCTTGAAATCGGTGAGCCCGTTATTGCCGCCGAAGCCCATGTCGTTGCGGAAGAAGGCGAGCAGCAGCGCATAGGTCATCGCCTGGGTGATGATCGAGAGATAGACCCCGTTGACCCGCGAGCGGAAGGCGAACCAGCCGAAGACGAAGGCAAGCAGGCCCGGCACCAGCACCACCATCAGCGCTGCGAACCAGAACATGTCGAAGCCGTACCAGAACCAGGGCAGCTCCTTCCAGTTGAGAAAGACCATGAAGTCCGGCAGCAGCGGATTGCCGTAGGAGCCGCGTGCGCCGATCTGGCGCATCAGATACATGCCCATGGCATAGCCGCCGAGCGCGAAGAAGGCCGCATGCCCCAGCGAGAGGATGCCGCAGAA"))

(defn convert-sequence [sequence]
  (let [data (frequencies
               (map 
                 convert-kmer
                 ((get-kmers k) sequence)))]
    (mat/sparse-array
      (map 
        (fn [x] 
          (get data x 0.0))
        (range 0 space)))))

; Convert into a sequence of 10 00bp
; This is for running the net, not training
(defn convert-sequences [seq-to-analyze]
  (let [small-seq (interleave
                    (range)
                    (map 
                      convert-sequence
                      (partition 1000 seq-to-analyze)))]
    small-seq))

(defrecord Training [data label])

(defn generate-training-set [seq-to-analyze label]
  (let [seqs (partition 
               1000
               1000 ; Can use smaller numbers to make a sliding window -- But file gets too big -- Need a way to store sparse matrices
               seq-to-analyze)]
      (map 
        (fn [x] (cons label (convert-sequence x)))
        seqs)))

(def categories 
  {:Main   [1.0 0.0 0.0 0.0 0.0]
   :pSymA  [0.0 1.0 0.0 0.0 0.0]
   :pSymB  [0.0 0.0 1.0 0.0 0.0]
   :others [0.0 0.0 0.0 1.0 0.0]
   :acc    [0.0 0.0 0.0 0.0 1.0]})

(defn create-training-set-from-fasta-file [file]
  (with-open [rdr (clojure.java.io/reader file)
              data (clojure.java.io/writer "data.tsv" :append true)]
    (doseq [line (apply concat
                   (map 
                     (fn [x]
                       (generate-training-set 
                         (clojure.string/upper-case (:seq x))
                         (keyword (:id x))))
                     (fasta/parse rdr)))]
      (.write data (clojure.string/join "\t" line))
      (.write data "\n"))))

; Tried to use btable but deps are too outdated...

(def params
  {:test-ds-size      5000
   :optimizer         (adam/adam)
   :batch-size        100
   :epoch-count       50
   :epoch-size        200000})

;(def network-description
;  [(layers/input 1000 1 1 :id :data)
;   (layers/convolutional space 0 1 20)
;   (layers/max-pooling 2 0 2)
;   (layers/dropout 0(defn- to-epoch-seq-fn  [item epoch-count]  (if-not (fn? item)    (parallel/create-next-item-fn     (let [retval (if (map? (first item))                    (repeat item)                    item)]       (if epoch-count         (take epoch-count retval)         retval)))    (create-n-callable-fn item epoch-count))).9)
;   (layers/relu)
;   (layers/convolutional 5 0 1 50)
;   (layers/max-pooling 2 0 2)
;   (layers/batch-normalization)
;   (layers/linear 1000)
;   (layers/relu :center-loss {:label-indexes {:stream :label}
;                              :label-inverse-counts {:stream :label}
;   :labels {:stream :labels}
;   :alpha 0.9
;   :lambda 1e-4
;   (layers/dropout 0.5)
;   (layers/linear 5)
;   (layers/softmax :id :label)})

(def nn
  (->
    [(layers/input space 1 1 :id :data)
     ; (layers/convolutional 5 0 1 5)
     ;     (layers/max-pooling 2 0 2)
     ;     (layers/dropout 0.9)
     ;     (layers/relu)
     ;     (layers/convolutional 5 0 1 20)
     ;     (layers/max-pooling 2 0 2)
     ;     (layers/batch-normalization)
     ;     (layers/linear 50)
     ;     (layers/dropout 0.5)
;     (layers/dropout 0.9)
     (layers/linear (Math/ceil (/ space 64)))
;     (layers/relu)
;     (layers/dropout 0.9)
     (layers/linear 4)
     (layers/softmax :id :label)]
   network/linear-network))

;(def train-orig2
;  (concat
;  (training-set seq-main [1.0 0.0 0.0])
;  (training-set seq-psyma [0.0 1.0 0.0])
;  (training-set seq-psymb [0.0 0.0 1.0]))

;(def train-test)
;  (concat))
;    (for [i (range 100)])))
;      {:data (take 5 (repeatedly #(rand-int 2)))})))
;       :label [1.0 0.0 0.0]})))
;    (for [i (range 100)])))
;      {:data (take 5 (repeatedly #(rand-int 2)))}))) 
;       :label [0.0 1.0 0.0]})))
;    (for [i (range 100)])))
;      {:data (take 5 (repeatedly #(rand-int 2)))}))) 
;       :label [0.0 0.0 1.0]})))

;(println train-test)

(def kh35c-main "GGGAGGTCGGTGCGCTTGGGGCCTACGGCTATCACGACGCCGTCGATTTCACGCCGACGCGCGTGCCGGAAGGCCAGAAATGCGCCGTCGTGCGCAACTATTATGCCCATCATCACGGCATGTCGGTCGCCGCGGTCGCCAATGTCGTCTTCAACGGGCAGCTGCGCGAGTGGTTCCACGCCGATCCCGTCATCGAGGCCGCCGAACTCCTCCTGCAGGAAAAGGCCCCGCGTGACATCCCGGTCATGGCAGCCAAGCGCGAGCCGGAAGCGCTGGGCAAGGGCCAGGCCGATCTCCTGCGCCCCGAAGTCCGCGTCGTCGAAGACCCGATCAATCAGGACCGCGAGACGGTGCTTCTGTCGAACGGTCACTACTCCGTCATGTTGACGGCGACAGGGGCGGGCTATGCCCGCTGGAACGGCCAGTCGGTCACGAGATGGACTCCGGACCCGGTAGAGGACAGGACGGGGACCTTCATCTTCCTTCGCGACACGGTGACGGGCGACTGGTGGTCGGCCACGGCCGAGCCCCGGCGTGCGCCGGGCGAAAAGACCGTTACCCGCTTCGGCGACGACAAGGCCGAATTCGTCAAGACCGTCGGCGATCTGACAAGCGAGGTGGAATGCATCGTCGCGACCGAGCACGATGCCGAAGGCCGCCGGGTTATCCTGCTCAACACGGGCACGGAAGACCGGTTCATCGAGGTGACCTCCTATGCCGAGCCGGTGCTTGCGATGGACGATGCCGACAGCTCGCACCCGACCTTCTCGAAGATGTTCCTGCGCACCGAGATCAGCCGTCACGGAGACGTGATATGGGTCTCGCGCAACAAGCGAAGCCCCGGCGATCCGGACATCGAGGTCGCCCATCTCGTCACCGACAATGCCGGCAGCGAGCGCCACACGCAGGCGGAAACCGATCGCCGGCGCTTCCTCGGCCAGGGCCGCACGCTTGCCGAGGCGGCCGCATTCGACCCGGGCGCCACGCTTTCCGGCACCGACGGCTTCACGCTCGATCCGATCGTGTCGCTCCGCCGCGTCGTACGCGTGCCGGCGGGCAAGAAAGTGAGCGTCATCTTCTGGACGATCGCCGCCCCGGACAGGGAAGGCGTCGACCGGGCGATCGACCGCTACCGGCATCCGGAAACCTTCAATCACGAGCTCATCCATGCCTGGACCCGCAGCCAGGTGCAGATGCGCCATGTCGGGATCACCTCGAAGGAGGCCGCGAGCTTCCAGATGCTCGGCCGCTATCTCGTCTATCCGGATATGCACCTTCGCGCCGACGCGGAGACCGTCAAGACCGGGCTCGCCTCGCAATCGGCGCTGTGGCCGCTGGCGATCTCCGGCGACTTCCCGATCTTCTGCCTCAGGATCAACGACGACGGCGATCTCGGCATCGCCCGCGAGGCCTTGCGGGCGCAGGAATATCTGAGAGCTCGCGGCATCACCGCCGATCTGGTGGTCGTCAACGAGCGCGCCTCCTCCTACGCGCAGGACCTGCAGCACACGCTCGACTCGATGTGCGAGAATTTGAGGCTTCGCGGCCTTTCGGACGGCCCGCGCCAGCACATATTTGCGGTGCGCCGGGACCTTATGGAACCGGAAACCTGGTCGACGCTGATCTCGGCATCCCGCGCCGTCTTCCATGCGCGCAACGGCACGATCTCGGATCAGATAGCCCGCGCCACATCGCTCTACTCCAAACCTTCCGAAAAGAAGGAGGAGGGCGCCGAGATGCTGCTGCCGGTGATACGGGAG")
(def wsm419-uni2 "CGTCAACGAGCTGCTCACCAACGCGCTCAAGCATGCTTTCAACGGCCGCGAAGGAGGAGTAATCACGCTGCGAAGCACTTTTGAGGATGATGGCTACCGTGTCATCGTTGCGGACGACGGAATAGGTTTCCCGGACGGAGAGACCTGGCCCAAACACGGCAAGCTTGGCGAGTTGATCGCGCAGTCGCTTCGCGAAAATTCCAGGGCTGATCTCCAGGTGATCTCCACGCCGGGTCAAGGCACACGCGCAACGATTCGTTTCCGGAACGACTCCGTATAGGCGCGCGAGCGTCCGAGCACCGCGCATCATAAGGCCGCGCGCGAACGCGGCTGCATAGGATGTCTATCGGCCCGGCTTATAGATCTGGTCGAAAATCCCCCATCGTCGAAGAACTTCGGCTGGGCTTCCTGCCAACCGCCGAAGTCGCCAATGGTGACCAGTTTGAGATCGGCAAAGCGTGCCGAGTCCGCGGGGTCGGCCAGCTCGGGCTTGAACGGCCGATAGTAGTGCTTGGCGACGATCTTCTGGCCGACGTCGCTGTAGAGGTAGCCGAGATAGGCTTCGGCAACATTGCGGGTGCCTTTGCTGTCGACATTCCCGTCCAAGAGCGCCACGGAGGGCTCGGCCCTGATCGATATGGACGGTGTCACGATCTCGAACTTGTCGGGGCCGAGTTCATCGAGCGCGAGATAGGCCTCATTCTCCCAGGCGAGCAGCACGTCGCCGAGCCCGCGATGGACGAAAGTGGTCATCGCTCCCCACGCGCCGGTGTCGAGAACGAGAACCTGCTTGAAAAGCGCCGCCGCATATTCCTGCGCCTTGGCCTCGTCGCCGTTGTTTGCATCCCGCGCCCAGGCCCAGGCTGCAAGGAAGTTCCAGCGCGCGCCACCCGAGGTCTT")

;println "Generating Training Set")

;(crit/quick-bench)
;  (create-training-set-from-fasta-file "data-files/Rm1021.final.fasta"))

;(def train-orig (create-training-set-from-fasta-file "data-files/Rm1021.final.fasta"))

;(println (first train-orig))
;(println (last train-orig))
;(println (count train-orig))

;(println (first train-orig))
;(println (reduce + (map mat/esum (map :data train-orig))))
;(println (map :label train-orig))

;(println train-orig2)

;(System/exit 0)


;(def training-data-file (iota/vec "data.tsv"))

; Can be called forever
;(defn train-orig [])
;   (let [r (clojure.string/split #"\t" (rand-nth training-data-file))]
;     (->Training (apply mat/sparse-array (rest r)) (get categories (first r) [0.0 0.0 0.0 1.0 0.0])))

(def train-orig [])

(defn -main []
  (let [[test-ds train-ds] (split-at 500 (shuffle train-orig))]
;(create-training-set-from-fasta-file "data-files/Rm1021.final.fasta")))]
    (try
      (def trained
        (experiment-train/train-n 
          nn
          train-ds
          test-ds
          :batch-size 500 :epoch-count 500))
      (catch Exception e
        (println e))))
  
  (println (execute/run trained [{:data (convert-sequence kh35c-main)}])))
                                 
