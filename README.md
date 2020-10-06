# Coming-to-Terms-Automatic-Formation-of-Neologisms-in-Hebrew

Code for the findings of EMNLP paper "Coming to Terms: Automatic Formation of Neologisms in Hebrew"

## The project is composed of the following folders:

combining_shoresh_mishkal_learning: contains all the necessary files (data + code) for training the seq2seq model (to modify the naive root and pattern combination into a valid Hebrew word).

crawler: 
* crawl.py: contains the code to crawl the Ben Yehuda project website for classics of Hebrew literature with diacritics (http://benyehuda.org/). 
* cleaner.py: contains the code to clean the crawled data from prefixes (מש"ה וכל"ב) according to grammar rules taken from the Hebrew Academy website (https://hebrew-academy.org.il/2013/07/18/%D7%A0%D7%99%D7%A7%D7%95%D7%93-%D7%90%D7%95%D7%AA%D7%99%D7%95%D7%AA-%D7%94%D7%A9%D7%99%D7%9E%D7%95%D7%A9/)
We used the cleaned crawled data to train an n-gram character-based language model. 

language_model: contains the code for training our language models (n-gram).

other: contains all the complementary code for running our Hebrew word formation scheme:
* word_generator.py: the core functions for the scheme to work 
* gzarot_tagging.py, naive_shoresh_mishkal_combine.py, seq2seq_shoresh_mishkal_concat.py: seq2seq model related functions
* wordnet_sister_terms.py: the functions to extract sister terms.
* utils.py: general functions.
* run_udpipe.py: functions for using the UDPipe POS tagger and dependency parser. 
* edit_distance.py: contains a function for niqqud normalization (for setting the distance of two diacritic characters that sound alike to zero)

eliezer.py: this file's main runs the whole scheme. Given an English word, returns Hebrew suggestions in a form of list of lists (the first list contains the root & pattern suggestions, the second the compound suggestions and the third the word blend suggestions) 

## To run the scheme one should:
* Download the Universal Dependencies 2.5 Models for UDPipe from: https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-3131 
* Copy the Hebrew model into the project, and load it in line 48 of eliezer.py
* Ask the missing data folder by contacting us via: stav.yardeni@cs.huji.ac.il, moranmiz@cs.huji.ac.il (we omitted the data due to its size and some restrictions on it).  

## Requirements: 
* matplotlib==3.3.0
* nltk==3.5 (download wordnet)
* numpy==1.19.1
* pandas==1.1.0
* ufal.udpipe==1.2.0.3
* pytorch==1.1.0

## Pointers to download the free sources we used:
#### English - English dictionaries:
(1) Wikipedia abstracts: https://dumps.wikimedia.org/enwiki/latest/ (we used the enwiki-latest-abstract.xml.gz file)  
(2) English Wiktionary: https://dumps.wikimedia.org/enwiktionary/latest/  
(3) Urban dictionary: https://github.com/mattbierner/urban-dictionary-entry-collector  
(4) Webster dictionary 1913: https://github.com/adambom/dictionary  
(5) WordNet definitions (via python's nltk)  
(6) Conceptnet definitions (via API)  
(7) Easy English student dictionary: https://www.easypacelearning.com/english-books/english-books-for-download-pdf/category/33-3-dictionaries-to-download-in-pdf  
#### English - Hebrew dictionaries:
(1) Hebrew Wiktionary: https://dumps.wikimedia.org/hewiktionary/latest/  
(2) English Wikitionary: https://dumps.wikimedia.org/enwiktionary/latest/  
(3) Hebrew WordNet: http://compling.hss.ntu.edu.sg/omw/  
(4) Wikipedia langlinks: https://dumps.wikimedia.org/hewiki/latest/ (we used the files: hewiki-latest-langlinks.sql.gz, hewiki-latest-page.sql.gz)  
#### Root and pattern source:
Hebrew Wiktionary: https://dumps.wikimedia.org/hewiktionary/latest/  
#### Synonyms:
(1) Hebrew Wiktionary: https://dumps.wikimedia.org/hewiktionary/latest/  
(2) Hebrew WordNet: http://cl.haifa.ac.il/projects/mwn/index.shtml  (we used the .sql files)  
#### Associations EAT database (for future work):
See https://joernhees.de/dump/papers/2016ESWCEATDBpedia.pdf (a link to their database is in the paper)  

## Have fun! 
