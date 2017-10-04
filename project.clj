(defproject nn-replicon-identification "0.1.0-SNAPSHOT"
  :description "FIXME: write description"
  :url "http://example.com/FIXME"
  :main nn-replicon-identification.core
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :plugins [[lein-gorilla "0.4.0"]]
  :dependencies [[org.clojure/clojure "1.8.0"]
                 [org.tensorflow/tensorflow "1.3.0"]
                 [net.mikera/core.matrix "0.61.0"]
                 [net.mikera/vectorz-clj "0.47.0"]
                 [org.clojure/math.numeric-tower "0.0.4"]
                 [clj-fuzzy "0.4.1"]
                 [thinktopic/cortex "0.9.21"]
                 [org.clojars.ds923y/nd4clj "0.1.1-SNAPSHOT"]
                 [thinktopic/experiment "0.9.21"]])
    
