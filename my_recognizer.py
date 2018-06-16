import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # implement the recognizer
    # return probabilities, guesses
	
    for test_word, (test_X, test_length) in test_set.get_all_Xlengths().items():
        b_score = float("-Inf")
        b_guess = ""
        prob = dict()
        for word, model in models.items():
            try:
                score = model.score(test_X, test_length)
            except Exception:
                score = float("-Inf")            
            if score > b_score:
                b_score = score
                b_guess = word
            prob[word] = score
        
        probabilities.append(prob)
        guesses.append(b_guess)
    return probabilities, guesses	
	
    raise NotImplementedError
