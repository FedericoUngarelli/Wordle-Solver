from random import choice
import yaml
from rich.console import Console
import re
from collections import Counter
import numpy as np
from itertools import permutations
import random
import pandas as pd

class Guesser:
    '''
        INSTRUCTIONS: This function should return your next guess. 
        Currently it picks a random word from wordlist and returns that.
        You will need to parse the output from Wordle:
        - If your guess contains that character in a different position, Wordle will return a '-' in that position.
        - If your guess does not contain thta character at all, Wordle will return a '+' in that position.
        - If you guesses the character placement correctly, Wordle will return the character. 

        You CANNOT just get the word from the Wordle class, obviously :)
    '''
    def __init__(self, manual):
        self.word_list = yaml.load(open('wordlist.yaml'), Loader=yaml.FullLoader)
        self._manual = manual
        self.console = Console()
        self._tried = []
        self.word_list_initial = self.word_list.copy()
        self.letters = 'abcdefghijklmnopqrstuvwxyz'
        self.dict = {key: self.letters for key in range(5)}
        self.guess = None

        ### For the starting word consider permutations (in most cases, they have higher expected info and the computation is much faster)
        ## Consider all the permutations of the first 8 most common letters in the english vocabulary
        self.permutations = [''.join(perm) for perm in permutations(['e', 'a', 'r', 'i', 's', 'n', 't'], 5)]
        # Get the 30 permutations with highest expected info
        top_30_permutations = self.list_to_df(self.compute_max_expected_info(self.permutation_perf_mat(self.permutations)), self.permutations).head(30)['words'].tolist()
        # Compute their 2-steps ahead expected info
        two_step_v = self.compute_max_expected_info2(self.permutation_perf_mat(top_30_permutations), top_30_permutations)
        # Take the one with highest 2-steps expected info
        self.best_starting = top_30_permutations[np.argmax(two_step_v)]
        
        ### PREVIOS VERSION (MUCH FASTER BUT LEES ACCURATE)
        #self.best_starting = self.permutations[self.compute_max_expected_info(self.permutation_perf_mat(self.permutations))]

    def restart_game(self):
        self._tried = []
        self.guess = None
        self.word_list = self.word_list_initial
        self.dict = {key: self.letters for key in range(5)}
    

    def get_guess(self, result):
        '''
        This function must return your guess as a string. 
        '''
        if self._manual=='manual':
            return self.console.input('Your guess:\n')
        else:
            '''
            CHANGE CODE HERE
            '''
            if self.guess is None:
                guess = self.best_starting                
                self.guess = guess
                self._tried.append(guess)
                self.console.print(guess)
                return guess
            
            else:
                words = ' '.join(self.word_list)
                matching_list = self.matching_words(words, self.guess, result)
                self.word_list = matching_list

                ### corner case: 1 '+' in the result
                if result.count('+') == 1 and result.count('-') == 0 and len(self._tried) < 6 and (2 < len(matching_list)):
                    diff = 5 - len(matching_list)
                    pos = result.index('+')
                    guess = (random.choice(self.dict[pos])) * diff
                    for i in range(min(5, len(matching_list))):
                        guess += matching_list[i][pos]
                    self.guess = guess
                    self._tried.append(guess)
                    self.console.print(guess)
                
                ### another corner case: 2 '+' in the result 
                elif result.count('+') == 2 and result.count('-') == 0 and len(self._tried) < 5 and (2 < len(matching_list)):
                    index1 = [i for i in range(len(result)) if result[i] == '+'][0]
                    index2 = [i for i in range(len(result)) if result[i] == '+'][1]
                    diff = 5
                    new_string = ['o', 'b', 'c', 'd', 'm'] 
                    for word in matching_list:
                        new_string.append(word[index1])
                        new_string.append(word[index2])
                    new_string.reverse()
                    new_string_l = list(set(new_string))[:5]
                    guess = ''.join(new_string_l)
                    self.guess = guess
                    self._tried.append(guess)
                    self.console.print(guess)
                    
                else:
                ### Entropy ###
                    patterns = self.compile_matrix2()
                    max_info_index = np.argmax(self.compute_max_expected_info(patterns))
                    guess = matching_list[max_info_index]
                    self.guess = guess
                    self._tried.append(guess)
                    self.console.print(guess)
                
                return guess

    def filter_generator(self, guess, result):
        "Return the filtering rules given a guess and a result"
        filter = self.dict
        filter2 = []
        for i in range(5):
            if result[i] in self.letters:
                filter[i] = result[i]
            elif result[i] == '+':
                for j in range(5):
                    if (result[j] not in self.letters) and (guess[i] not in filter2):
                        filter[j] = filter[j].replace(guess[i],'')       
            elif result[i] == '-':
                filter[i] = filter[i].replace(guess[i],'')
                filter2.append(guess[i])
        return filter, filter2
    
    def generate_regex(self, dictionary):
        """ Builds patter given the dictionary """
        regex_pattern = ''
        for key, value in dictionary.items():
            regex_pattern += '[' + value + ']'
        return regex_pattern

    def generate_regex2(self, letters):
        """ Builds patter of letters that must be included """
        regex_pattern = '^'
        for letter in letters:
            regex_pattern += '(?=.*' + re.escape(letter) + ')'
        regex_pattern += '[a-zA-Z]+$'
        return regex_pattern
    
    def matching_words(self, words, guess, result):
        x =  self.filter_generator(guess, result)[0]
        matching_words_first = re.findall(self.generate_regex(x), words)
        y = self.filter_generator(guess, result)[1]
        matching_words_second = [word for word in matching_words_first if re.match(self.generate_regex2(y), word)]
        return matching_words_second


    """Implement selection based on entropy"""
    
    def compile_matrix2(self):
        n = len(self.word_list)
        matches = [self.get_matches(word2, word1) for word1 in self.word_list for word2 in self.word_list]
        matches = np.array(matches).reshape(n, n)
        return matches
    

    def get_matches(self, word1, word2):
        #### word2 is the guessed word, word1 is the right word    
        counts = Counter(word1)
        results = []
        for char1, char2 in zip(word1, word2):
            if char1 == char2:
                results.append(char2)
                counts[char2] -=1
            else:
                results.append('+')
        for i, char2 in enumerate(word2):
            if word2[i] != word1[i] and word2[i] != word1[i] and counts[char2] > 0:
                counts[char2] -= 1
                results[i] = '-'
        return ''.join(results)
    
    def compute_max_expected_info(self, matrix):
        expected_info_l = []
        total_strings = matrix.shape[1]
        for row in matrix:
            string_counts = Counter(row)
            probabilities = np.array(list(string_counts.values())) / total_strings
            quantities = np.log2(1 / probabilities)
            expected_info = np.sum(probabilities * quantities)
            expected_info_l.append(expected_info)        
        expected_info_arr = np.array(expected_info_l)
        return expected_info_arr
    
    def permutation_perf_mat(self, permutations):
        ''' computes pattern matrix for all possible permutations '''
        m = len(permutations)
        n = len(self.word_list)
        matches = [self.get_matches(word2, word1) for word1 in permutations for word2 in self.word_list]
        matches = np.array(matches).reshape(m, n)
        return matches

    def list_to_df(self, vector, word_list):
        ''' useful to sort lists from a dataframe '''
        data = {
                'entropy': vector,
                'words': word_list
                }
        # Create DataFrame
        df = pd.DataFrame(data)
        df = df.sort_values(by='entropy', ascending=False)
        return df
    
    def compute_max_expected_info2(self, matrix, guessed_words):
        ''' computes 2-steps ahead entropy '''
        word_list = ' '.join(self.word_list)
        expected_info_l = []
        two_step_expected_1 = []
        total_strings = matrix.shape[1]
        self.dict = {key: self.letters for key in range(5)}
        for i in range(len(matrix)):
            string_counts = Counter(matrix[i])
            probabilities = np.array(list(string_counts.values())) / total_strings
            quantities = np.log2(1 / probabilities)
            expected_info = np.sum(probabilities * quantities)
            expected_info_l.append(expected_info)        
            two_step_expected_2 = 0
            for pattern, counter in string_counts.items():
                self.dict = {key: self.letters for key in range(5)}
                matching_words_l = self.matching_words(word_list, guessed_words[i], pattern)
                n = len(matching_words_l)
                matches = [self.get_matches(word2, word1) for word1 in matching_words_l for word2 in matching_words_l]
                submatrix = np.array(matches).reshape(n, n)
                expected_info_sub_l = []
                total_string_sub = submatrix.shape[1]
                for row in submatrix:
                    string_counts_sub = Counter(row)
                    probabilities_sub = np.array(list(string_counts_sub.values())) / total_string_sub
                    quantities_sub = np.log2(1 / probabilities_sub)
                    expected_info_sub = np.sum(probabilities_sub * quantities_sub)
                    expected_info_sub_l.append(expected_info_sub)        
                max_expected_info_sub = max(expected_info_sub_l) if submatrix.shape[1] > 0 else 0
                two_step_expected_2 += (max_expected_info_sub * counter / total_strings)

            two_step_expected_1.append((two_step_expected_2))
        two_step_expected_arr = np.array(two_step_expected_1)
        expected_info_arr = np.array(expected_info_l)
        final_vector = expected_info_arr + two_step_expected_arr
        
        return final_vector
