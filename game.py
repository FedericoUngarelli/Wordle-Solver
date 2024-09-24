import sys
from wordle import Wordle
from guesser import Guesser
import argparse


class Game:
    global RESULTS, GUESSES
    GUESSES = [] # number of guesses per game
    RESULTS = [] # was the word guessed?
        
    def score(result, guesses):
        if '-' in result or '+' in result:
            # word was not guessed
            result = False
        else:
            result = True
        RESULTS.append(result)
        if result:
            GUESSES.append(guesses)



    def game(wordle, guesser):
        endgame = False
        guesses = 0
        result = '+++++'
        while not endgame:
            guess = guesser.get_guess(result)
            guesses+=1
            result, endgame = wordle.check_guess(guess)    
            print(result)
        return result, guesses
            
            
    if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument('--r', type=int)
        args = parser.parse_args()
        if args.r:
            successes = []
            wordle = Wordle()
            guesser = Guesser('console')
            for run in range(args.r):
                guesser.restart_game()
                wordle.restart_game()
                results, guesses = game(wordle, guesser)
                score(results, guesses)
            print("You correctly guessed {}% of words. ".format( RESULTS.count(True)/len(RESULTS)*100))
            if GUESSES:
                print("Average number of guesses: ", sum(GUESSES)/len(GUESSES))
        else:
            # Play manually on console
            guesser = Guesser('manual')
            wordle = Wordle()
            print('Welcome! Let\'s play wordle! ')
            game(wordle, guesser)
        