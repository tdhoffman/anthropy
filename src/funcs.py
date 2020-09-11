import numpy as np
import pandas as pd

def bayes_consensus(answersGiven, competancy, numAnswers, prior=-1):
    if prior == -1:
        prior = (1/numAnswers)*np.ones((numAnswers, answersGiven.shape[1]))
  
    if not np.all(np.abs(prior.sum(axis=1) - 1) < 0.001):
        raise RuntimeError('Your Prior for every question must add to 1.')    
  
    if not np.all(prior >= 0): 
        raise RuntimeError('Your Prior Must assign non-negative probability to all possible outcomes (preferably positive).')
  
    if (prior.shape[1] != answersGiven.shape[1] or prior.shape[0] != numAnswers):
        raise RuntimeError('Your Prior matrix is wrong shape.')    
  
    probability = pd.DataFrame(np.zeros((numAnswers, answersGiven.shape[1])), columns=list(answersGiven))
    wrong = (1-competancy)*(numAnswers-1)/numAnswers
    right = 1 - wrong
    for QQQ in range(1, answersGiven.shape[1]):
        for aaa in range(1, numAnswers):
            probability[aaa,QQQ] = np.prod(np.where(answersGiven[:,QQQ] == aaa, right, wrong)) 
        probability[:,QQQ] = probability[:,QQQ]*prior[:,QQQ]    
        probability[:,QQQ] = probability[:,QQQ]/sum(probability[:,QQQ])    
      
    return probability

def discounted_agreement_matrix(surveyResults, numAns):
    output = np.zeros((surveyResults.shape[0], surveyResults.shape[0]))
    for iii in range(1, output):
        for jjj in range(1, output):
            output[iii,jjj] = sum(surveyResults[iii,:] == surveyResults[jjj,:])/surveyResults.shape[1]      
                 
    output = (output*numAns - 1)/(numAns-1)
    np.fill_diagonal(output, 0)
  
    return output