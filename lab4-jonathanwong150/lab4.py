import numpy as np
from lab4_utils import feature_names
from collections import defaultdict

# Hint: Consider how to utilize np.unique()
def preprocess_data(training_inputs, testing_inputs, training_labels, testing_labels):
    processed_training_inputs, processed_testing_inputs = ([], [])
    processed_training_labels, processed_testing_labels = ([], [])
    # VVVVV YOUR CODE GOES HERE VVVVV $
    # Swap row and columns of training_inputs
    trainingT = np.transpose(training_inputs)
    # Iterate through rows
    for i in range(len(trainingT)):
        # Use np.unique to get array of unique features and their occurences
        feature, occurence = np.unique(trainingT[i], return_counts = True)
        # Calculate the mode 
        mode = feature[np.where(occurence==max(occurence))[0]]
        # Iterate through the columns of transposed training matrix
        for j in range(len(trainingT[i])):
            # If a value is missing, replace with the mode
            if trainingT[i][j] == '?':
                trainingT[i][j] = mode
    # Tranpose test set
    testingT = np.transpose(testing_inputs)
    # Iterate through rows of test set
    for i in range(len(testingT)):
        # Find array of unique features and their occurrences
        feature, occurence = np.unique(testingT[i], return_counts = True)
        # Find mode of each feature
        mode = feature[np.where(occurence==max(occurence))[0]]
        # Iterate through columns and replace missing colummns with the mode.
        for j in range(len(testingT[i])):
            if testingT[i][j] == '?':
                testingT[i][j] = mode
    
    # Clean strange inconsistencies
    for i in range(len(trainingT)):
        for j in range(len(trainingT[i])):
            if trainingT[i][j]==['no']:
                trainingT[i][j]='no'
            if trainingT[i][j]==['left_low']:
                trainingT[i][j]='left_low'

    finalTrainingInput = np.transpose(trainingT)
    finalTestingInput = np.transpose(testingT)
    trainingLabel = training_labels
    testingLabel = testing_labels
    # ^^^^^ YOUR CODE GOES HERE ^^^^^ $
    return finalTrainingInput, finalTestingInput, trainingLabel, testingLabel



# Hint: consider how to utilize np.count_nonzero()
def naive_bayes(training_inputs, testing_inputs, training_labels, testing_labels):
    assert len(training_inputs) > 0, f"parameter training_inputs needs to be of length 0 or greater"
    assert len(testing_inputs) > 0, f"parameter testing_inputs needs to be of length 0 or greater"
    assert len(training_labels) > 0, f"parameter training_labels needs to be of length 0 or greater"
    assert len(testing_labels) > 0, f"parameter testing_labels needs to be of length 0 or greater"
    assert len(training_inputs) == len(training_labels), f"training_inputs and training_labels need to be the same length"
    assert len(testing_inputs) == len(testing_labels), f"testing_inputs and testing_labels need to be the same length"
    misclassify_rate = 0
    # VVVVV YOUR CODE GOES HERE VVVVV $
    # Iterate through all of the testing inputs/patients
    for i in range(len(testing_inputs)):
        recurProb = 1
        noRecurProb = 1
        # Iterate through each feature of a patient
        for j in range(len(testing_inputs[0])):
            currVal = testing_inputs[i][j]
            # Find the indices of the training points that have the same value for the jth feature
            sameValueIndices = np.where(training_inputs[:, j] == currVal)[0]
            sameValueCount = len(sameValueIndices)
            # Find the number of training points that have the same value for the jth feature and a recurrence/no-recurrence
            recurrenceCount = len(np.where(training_labels[sameValueIndices] == 'recurrence-events')[0])
            noRecurrenceCount = len(np.where(training_labels[sameValueIndices] == 'no-recurrence-events')[0])
            # Laplace smoothing and calculate posterior probabilities
            uniqueVals = len(np.unique(training_inputs[:,j]))
            recurProb *= (recurrenceCount + 1) / (sameValueCount + uniqueVals)
            noRecurProb *= (noRecurrenceCount + 1) / (sameValueCount + uniqueVals)
        # Compare probabilities and determine label
        if recurProb > noRecurProb:
            predicted_label = 'recurrence-events'
        else:
            predicted_label = 'no-recurrence-events'
        # If label is incorrect, increment misclassify_rate
        if predicted_label != testing_labels[i]:
            misclassify_rate += 1
    # Determine misclasify_rate
    misclassify_rate /= len(testing_labels)
    # ^^^^^ YOUR CODE GOES HERE ^^^^^ $
    return misclassify_rate


# Hint: reuse naive_bayes to compute the misclassification rate for each fold.
def cross_validation(training_inputs, testing_inputs, training_labels, testing_labels):
    data = np.concatenate((training_inputs, testing_inputs))
    label = np.concatenate((training_labels, testing_labels))
    average_rate = 0
    # VVVVV YOUR CODE GOES HERE VVVVV $

    # VVVVV YOUR CODE GOES HERE VVVVV $
    return average_rate
