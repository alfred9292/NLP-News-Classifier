# NLP-News-Classifier

NLP Logistic Regression Implementation to classify news into 10 categories.

exp0.py: baseline system, only aplly bag-of-words, no further feature extraction performed, use a sparse representation of a feature vector.

exp1.py: TF-IDF weighting methodology used instead of bag-of-words

exp2.py: N-gram specification, only trains the title of news

exp3.py: N-gram specification, combine title and content as feature vector for training and testing

exp4.py: Normalization and Weighting, stop word list implemented

exp5.py: Syntax and Stylometry, use spaCY to find named entity. intead of using title or content, the experiment only considers named entity for classification.\
