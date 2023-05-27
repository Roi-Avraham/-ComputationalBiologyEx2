# -ComputationalBiologyEx2
This exercise has been writen by Sigal grabois and Roi Avraham.
# Description
For this assignment we were asked to decipher a mono-alphabetic cipher, this is a cipher built by replacing one letter with another.
The input files received for the purpose of the exercise and required to be in the work folder:
- An enc.txt file containing a text segment containing words encrypted by a mono-alphabetic cipher that must be decoded.
- A file named dict.txt containing a "dictionary" of common English words, a large majority of the words in the text (and of course many other words) are found in it.
- A file called LetterFreq.txt that contains the frequency of the letters of the English alphabet.
- A file called Letter2Freq.txt that contains the frequency of letter pairs (the statistics were calculated from a large database of English texts).

# The output files:
- A file called plain.txt that will contain the encrypted text in decrypted form
- A file named perm.txt that will contain for each character its permutation.<br>

The algorithm output 2 graphs:
![img.png](img.png)

![img_1.png](img_1.png)

# Installation
In order to run this exercise, you have 2 options:
<br> 1) Double-click on the Genetic_algorithm file.
<br> 2) You can run the Genetic_algorithm.py file by python Genetic_algorithm.py .
In order to do so you will need to install by pip install the next libraries:
* random
* string
* tkinter
* statistics
* numpy
* matplotlib
<br>


# Usage
After running the program (by anyone of the options, we tell in the Installation section),
the next window will show up:
<br>
![img_2.png](img_2.png)

In this window, you will need to enter 2 things:
1) In the first textbox you need to enter the run mode of the algorithm. The run mode can be only <b>one of
the next inputs</b>:
- C 
- D 
- L 
<br>
If you want to run the algorithm in the classic mode, so you will need to enter C. <br>
If you want to run the algorithm in the darwin mode, so you will need to enter D. <br>
If you want to run the algorithm in the lamark mode, so you will need to enter L. <br>
Please note that if you will not enter one of these input, this error message will show on the console:
<br>
![img_3.png](img_3.png)
<br>
and the program will finish.

2) In the second textbox, you will need to enter the path for the encrypted file to be decrypted.
Please note that if you will enter incorrect path, you will show the next error in the console:
![img_4.png](img_4.png)

- After entering the inputs, click on the submit button and the algorithm will start to run.
  the mode run will be print to the console and after each generation, data about the generation will be print for example:
  ![img_5.png](img_5.png)
- When the algorithm finishes, you will see the 2 graphs and the files, we mentioned before.

# Dictionary
Genetic_algorithm.py - Document that containing the genetic algorithm for decipher a mono-alphabetic cipher.
Genetic_algorithm.exe - the exe file of Genetic_algorithm.