# -ComputationalBiologyEx2
This exercise has been writen by Sigal grabois and Roi Avraham.
# Description
For this assignment we were asked to decipher a mono-alphabetic cipher, this is a cipher built by replacing one letter with another.
The input files received for the purpose of the exercise and required to be in the work folder:
- An enc.txt file containing a text segment containing words encrypted by a mono-alphabetic cipher that must be decoded.
- A file named dict.txt containing a "dictionary" of common English words, a large majority of the words in the text (and of course many other words) are found in it.
- A file called LetterFreq.txt that contains the frequency of the letters of the English alphabet.
- A file called Letter2Freq.txt that contains the frequency of letter pairs (the statistics were calculated from a large database of English texts).<br>
The output files are:
- A file called plain.txt that will contain the encrypted text in decrypted form
- A file named perm.txt that will contain for each character its permutation.

# Installation
in order to run this exercise, You can run the Genetic_algorithm.py file. 
in order to do so you will need the install by pip install the next libaries:
* random
* string
* sys
* statistics
* numpy
* matplotlib

# Usage
In order to run the exercise, we will enter the parameters so that when we run the program, we will pass as a parameter 
one of the parameters C L D, with each indicating the type of run, classic (C), Lamark (L) or Darwinian (D) respectively.
For example, in order to run the Darwinian version, we must run the py file and pass the parameter to it as follows: 
python Genetic_algorithm.py D
We set the parameters in the code as follows: (you set these in the Genetic_algorithm.py in lines 30-34)
POPULATION_SIZE The size of the population
NUM_GENERATIONS The maximum number of generations to run
MUTATION_RATE Mutation rate
ELITE_SIZE The size of the elite population that we transfer to the next generation without making any changes to it.
TOURNAMENT_SIZE Population size for competition and crossover.

# Dictionary
Genetic_algorithm - Document that containing the gentic algorithm for decipher a mono-alphabetic cipher.
