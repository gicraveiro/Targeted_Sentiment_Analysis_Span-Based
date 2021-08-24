Final Project - Sentiment Analysis 3 
Targeted Sentiment Analysis via Span-based Extraction and Classification
Course: Natural Language Understanding
University of Trento
02/21 - 09/21

This is the repository for the final project of the NLU course. It contains:
-> a data folder with the dataset SemEval2014, which is composed of a train and a test file, annotated on laptop domain.
-> A report that details the process behind the code.
-> The code for the proposed tasks, in file main.py

Instructions:

To run the code, clone the repository and use command:

python3 main.py

To run target extraction, enter 0

To run polarity classification, enter 1

Each option takes approximately 17 minutes to run

It might also be necessary to install some libraries, if not present, run:

pip3 install transformers
pip3 install pytorch

If some library is not recognized by the system, try the general formula pip3 install name_of_the_library
These commands are meant for Python 3, and pip 3, for previous versions try removing the "3" as in "python main.py" and "pip install"

Inside the other branches ( not the master) and possibly in the local repository are traces of the roads not taken.

Additional remarks:
Results are poor, code must go through revision 
Potential issues are:
How to deal with sentences, spans and tokens that are not/ do not have targets/polarities
How to deal with results from the models and calculate the measures properly
Model configurations
Uncatched coding error
