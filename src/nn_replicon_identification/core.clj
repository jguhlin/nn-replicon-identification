(ns nn-replicon-identification.core
  (:require [clojure.core.matrix :as mat]
            [clojure.core.matrix.linear :as mat-linear]
            [clojure.core.reducers :as r]
            [cortex.experiment.train :as train]
            [cortex.nn.layers :as layers]
            [cortex.nn.network :as network]
            [cortex.nn.execute :as execute]
            [cortex.optimize.adadelta :as adadelta]
            [cortex.optimize.adam :as adam]
            [cortex.metrics :as metrics]
            [cortex.util :as util]
            [iota :as iota]
            [cortex.optimize.adam :as adam]
            [cortex.experiment.train :as experiment-train]
            [cortex.nn.execute :as execute]
            [clojure.math.numeric-tower :as math]
            [biotools.fasta :as fasta]))

(mat/set-current-implementation :vectorz)

; k of 11 is TOO BIG for nn to for
; k of 9 is too big
(def k 5)
(def space (math/expt 5 k))

; Use base 5, 0 is \N, etc...
;(Integer/toString 1953124 5)
; Build sparse-array 

(defn get-kmers [k]
  (fn [sequence]
    (distinct
      (partition k 1 sequence)))) ; Could move sliding window by k, or could do 1

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

(def kh35c-main 
  (clojure.string/upper-case "GGGAGGTCGGTGCGCTTGGGGCCTACGGCTATCACGACGCCGTCGATTTCACGCCGACGCGCGTGCCGGAAGGCCAGAAATGCGCCGTCGTGCGCAACTATTATGCCCATCATCACGGCATGTCGGTCGCCGCGGTCGCCAATGTCGTCTTCAACGGGCAGCTGCGCGAGTGGTTCCACGCCGATCCCGTCATCGAGGCCGCCGAACTCCTCCTGCAGGAAAAGGCCCCGCGTGACATCCCGGTCATGGCAGCCAAGCGCGAGCCGGAAGCGCTGGGCAAGGGCCAGGCCGATCTCCTGCGCCCCGAAGTCCGCGTCGTCGAAGACCCGATCAATCAGGACCGCGAGACGGTGCTTCTGTCGAACGGTCACTACTCCGTCATGTTGACGGCGACAGGGGCGGGCTATGCCCGCTGGAACGGCCAGTCGGTCACGAGATGGACTCCGGACCCGGTAGAGGACAGGACGGGGACCTTCATCTTCCTTCGCGACACGGTGACGGGCGACTGGTGGTCGGCCACGGCCGAGCCCCGGCGTGCGCCGGGCGAAAAGACCGTTACCCGCTTCGGCGACGACAAGGCCGAATTCGTCAAGACCGTCGGCGATCTGACAAGCGAGGTGGAATGCATCGTCGCGACCGAGCACGATGCCGAAGGCCGCCGGGTTATCCTGCTCAACACGGGCACGGAAGACCGGTTCATCGAGGTGACCTCCTATGCCGAGCCGGTGCTTGCGATGGACGATGCCGACAGCTCGCACCCGACCTTCTCGAAGATGTTCCTGCGCACCGAGATCAGCCGTCACGGAGACGTGATATGGGTCTCGCGCAACAAGCGAAGCCCCGGCGATCCGGACATCGAGGTCGCCCATCTCGTCACCGACAATGCCGGCAGCGAGCGCCACACGCAGGCGGAAACCGATCGCCGGCGCTTCCTCGGCCAGGGCCGCACGCTTGCCGAGGCGGCCGCATTCGACCCGGGCGCCACGCTTTCCGGCACCGACGGCTTCACGCTCGATCCGATCGTGTCGCTCCGCCGCGTCGTACGCGTGCCGGCGGGCAAGAAAGTGAGCGTCATCTTCTGGACGATCGCCGCCCCGGACAGGGAAGGCGTCGACCGGGCGATCGACCGCTACCGGCATCCGGAAACCTTCAATCACGAGCTCATCCATGCCTGGACCCGCAGCCAGGTGCAGATGCGCCATGTCGGGATCACCTCGAAGGAGGCCGCGAGCTTCCAGATGCTCGGCCGCTATCTCGTCTATCCGGATATGCACCTTCGCGCCGACGCGGAGACCGTCAAGACCGGGCTCGCCTCGCAATCGGCGCTGTGGCCGCTGGCGATCTCCGGCGACTTCCCGATCTTCTGCCTCAGGATCAACGACGACGGCGATCTCGGCATCGCCCGCGAGGCCTTGCGGGCGCAGGAATATCTGAGAGCTCGCGGCATCACCGCCGATCTGGTGGTCGTCAACGAGCGCGCCTCCTCCTACGCGCAGGACCTGCAGCACACGCTCGACTCGATGTGCGAGAATTTGAGGCTTCGCGGCCTTTCGGACGGCCCGCGCCAGCACATATTTGCGGTGCGCCGGGACCTTATGGAACCGGAAACCTGGTCGACGCTGATCTCGGCATCCCGCGCCGTCTTCCATGCGCGCAACGGCACGATCTCGGATCAGATAGCCCGCGCCACATCGCTCTACTCCAAACCTTCCGAAAAGAAGGAGGAGGGCGCCGAGATGCTGCTGCCGGTGATACGGGAG"))
(def wsm419-uni2 
  (clojure.string/upper-case "CGTCAACGAGCTGCTCACCAACGCGCTCAAGCATGCTTTCAACGGCCGCGAAGGAGGAGTAATCACGCTGCGAAGCACTTTTGAGGATGATGGCTACCGTGTCATCGTTGCGGACGACGGAATAGGTTTCCCGGACGGAGAGACCTGGCCCAAACACGGCAAGCTTGGCGAGTTGATCGCGCAGTCGCTTCGCGAAAATTCCAGGGCTGATCTCCAGGTGATCTCCACGCCGGGTCAAGGCACACGCGCAACGATTCGTTTCCGGAACGACTCCGTATAGGCGCGCGAGCGTCCGAGCACCGCGCATCATAAGGCCGCGCGCGAACGCGGCTGCATAGGATGTCTATCGGCCCGGCTTATAGATCTGGTCGAAAATCCCCCATCGTCGAAGAACTTCGGCTGGGCTTCCTGCCAACCGCCGAAGTCGCCAATGGTGACCAGTTTGAGATCGGCAAAGCGTGCCGAGTCCGCGGGGTCGGCCAGCTCGGGCTTGAACGGCCGATAGTAGTGCTTGGCGACGATCTTCTGGCCGACGTCGCTGTAGAGGTAGCCGAGATAGGCTTCGGCAACATTGCGGGTGCCTTTGCTGTCGACATTCCCGTCCAAGAGCGCCACGGAGGGCTCGGCCCTGATCGATATGGACGGTGTCACGATCTCGAACTTGTCGGGGCCGAGTTCATCGAGCGCGAGATAGGCCTCATTCTCCCAGGCGAGCAGCACGTCGCCGAGCCCGCGATGGACGAAAGTGGTCATCGCTCCCCACGCGCCGGTGTCGAGAACGAGAACCTGCTTGAAAAGCGCCGCCGCATATTCCTGCGCCTTGGCCTCGTCGCCGTTGTTTGCATCCCGCGCCCAGGCCCAGGCTGCAAGGAAGTTCCAGCGCGCGCCACCCGAGGTCTT"))


(defn convert-sequence [sequence]
  (let [mfactor (/ 1000 (count sequence))
        data (frequencies
               (map 
                 convert-kmer
                 ((get-kmers k) sequence)))]
    (mat/sparse-array
      (map 
        (fn [x] 
          (* mfactor (get data x 0.0)))
        (range 0 space)))))

; Convert into a sequence of 10 00bp
; This is for running the net, not training
(defn convert-sequences [seq-to-analyze]
  (let [small-seq (interleave
                    (range)
                    (map 
                      convert-sequence
                      (partition-all 1000 seq-to-analyze)))]
    small-seq))

(defrecord Training [data label])

;(let [data (sort-by key (frequencies (map convert-kmer ((get-kmers 5) seq1))))])
;      indices (map first data)])
;      vals (map second data)])
;  (mat/set-indices (mat/new-sparse-array space) indices vals))

(defn convert-sequence-training [sequence]
  (let [mfactor (/ 1000 (count sequence))
        data (sort-by 
               key
               (frequencies
                 (map 
                   convert-kmer
                   ((get-kmers k) sequence))))]
    (flatten
      (for [[k v] data]
        [(.toString (biginteger k) 36) (float (* mfactor v))]))))

(defn generate-training-set [seq-to-analyze label]
  (let [seqs (concat
               (partition-all 
                 1000
                 1000 ; Can use smaller numbers to make a sliding window -- But file gets too big -- Need a way to store sparse matrices
                 seq-to-analyze)
               (partition-all ; Vary this number so we can train on non-whole numbers
                 500
                 500
                 seq-to-analyze)
               (partition-all ; Train on larger sets of sequence
                 2000
                 2000
                 seq-to-analyze))]
      (map 
        (fn [x] (cons label (convert-sequence-training x)))
        seqs)))

(def categories 
  {:Main   [1.0 0.0 0.0 0.0 0.0]
   :pSymA  [0.0 1.0 0.0 0.0 0.0]
   :pSymB  [0.0 0.0 1.0 0.0 0.0]
   :others [0.0 0.0 0.0 1.0 0.0]
   :acc    [0.0 0.0 0.0 0.0 1.0]})

(defn -parse-seq [s]
  (if (= 0.0 (mat/esum s)) 
    (str "A" (.toString (biginteger (first (mat/shape s))) 36))
    (clojure.string/join "\t" s)))
  
(defn serialize-sparse [a]
  (clojure.string/join "\t" (map -parse-seq (partition-by zero? a))))

(defn -unparse-seq [s]
  (if (= \A (first s))
    (repeat (read-string (new BigInteger (apply str (rest s)) 36) 0.0))
    (Double/parseDouble s)))

(defn deserialize-sparse [s]
  (mat/sparse-array (flatten (map -unparse-seq (clojure.string/split s #"\t")))))

(defn convert-to-training [line]
  (let [ds (deserialize-sparse line)]
    {:data (mat/sparse-array (drop (count (first categories)))) :labels (take 5 ds)}))
  
(defn create-training-set-from-fasta-file [file]
  (with-open [rdr (clojure.java.io/reader file)
              data (clojure.java.io/writer "data.atsv" :append true)]
    (doseq [lines (map 
                    (fn [x]
                     (generate-training-set 
                       (clojure.string/upper-case (:seq x))
                       (keyword (:id x))))
                    (fasta/parse rdr))]
      (doseq [line lines]
        (.write data (clojure.string/join "\t" (concat line)))
        (.write data "\n")))))
; (create-training-set-from-fasta-file "data-files/Rm1021.final.fasta")

(defn parse-line [line]
  (let [line-p (clojure.string/split line #"\t")
        category (read-string (first line-p))
        data (partition 2 (rest line-p))]
   (->Training
     (mat/set-indices 
       (mat/new-sparse-array space) 
       (map (comp (fn [x] (new BigInteger x 36)) first) data) 
       (map (comp read-string second) data))
     (get categories category))))

(defn get-training-item [batch-size]
  (let [training-file (iota/vec "data.atsv")]
    (fn []
      (repeatedly batch-size #(parse-line (rand-nth training-file))))))

; This makes very large files (>20Gb for a single strain)
(defn create-training-set-from-fasta-file-to-tsv [file]
  (with-open [rdr (clojure.java.io/reader file)
              data (clojure.java.io/writer "data.tsv" :append true)]
    (doseq [line (pmap 
                   (fn [x]
                     (generate-training-set 
                       (clojure.string/upper-case (:seq x))
                       (keyword (:id x))))
                   (fasta/parse rdr))]
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


;         (layers/linear (* 10 (Math/ceil (math/sqrt (math/sqrt space)))))
         ;         (layers/convolutional 20 0 1 5)
;         (layers/dropout 0.9)
;         (layers/linear space)
;         (layers/linear 2048)
         ;         (layers/relu)
 ;         (layers/linear->softmax 5)

(defn define-nn []
  (def nn
    (->
      [(layers/input space 1 1 :id :data)
       (layers/linear 256)
       ;(layers/relu)
       (layers/linear 5)
       (layers/softmax :id :label)]
     network/linear-network)))

(defn train []
  (let [data-fn (get-training-item 2000)]
   (try
     (def trained
       (experiment-train/train-n 
         nn
         data-fn
         data-fn
         :batch-size 2000 :epoch-count 500))
     (catch Exception e
       (println e))))
  
  (println (execute/run trained [{:data (convert-sequence kh35c-main)}])))


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

;(def train-orig [])

(defn -main [])
;  (let [[test-ds train-ds] (split-at 500 (shuffle train-orig))]
  ;(create-training-set-from-fasta-file "data-files/Rm1021.final.fasta")))]
;  (try
;  (def trained
;  (experiment-train/train-n 
;  nn
;  train-ds
;  test-ds
;  :batch-size 500 :epoch-count 500
;  (catch Exception e
;  (println e)
  
;  (println (execute/run trained [{:data (convert-sequence kh35c-main)}])))
                                 

