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
* Ask the missing data folder by contacting us via: (<mails>) (we omitted the data due to its size and some restrictions on it).  

## Requirements: 
* matplotlib==3.3.0
* nltk==3.5 (download wordnet)
* numpy==1.19.1
* pandas==1.1.0
* ufal.udpipe==1.2.0.3
* pytorch==1.1.0

## Have fun! 
