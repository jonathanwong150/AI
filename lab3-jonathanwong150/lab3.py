import numpy as np
from lab3_utils import edit_distance, feature_names

# Hint: Consider how to utilize np.unique()
def preprocess_data(training_inputs, testing_inputs, training_labels, testing_labels):
    finalTrainingInput, finalTestingInput = ([], [])
    trainingLabel, testingLabel = ([], [])
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

    # Create dictionary that holds features and their groupings. Mapped by index
    features = {
    0: ['10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89', '90-99'],
    2: ['0-4', '5-9', '10-14', '15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59'],
    3: ['0-2', '3-5', '6-8', '9-11', '12-14', '15-17', '18-20', '21-23', '24-26', '27-29', '30-32', '33-35', '36-39'],
    4: ['no', 'yes'],
    5: [1, 2, 3],
    6: ['left', 'right'],
    8: ['no','yes']
    }
    # Create dictionaries for one hot encoding
    meno = {
        'lt40': [1,0,0],
        'ge40':[0,1,0],
        'premeno': [0,0,1]
    }
    quad = {
        'left_up': [1,0,0,0,0],
        'left_low':[0,1,0,0,0],
        'right_up':[0,0,1,0,0],
        'right_low':[0,0,0,1,0],
        'central':[0,0,0,0,1]
    }
    # Store two lists for the variables we will need to one-hot encode
    menoAddCols = []
    quadAddCols = []
    # Iterate through each row of the training data
    for i in range(len(trainingT)):
        encode = []
        # If i represents a feature that does not need to be one-hot encoded
        if i in features:
            encode = features[i]
            # Iterate through the columns and changed all of the values to the index it exists in the encoder
            for j in range(len(trainingT[i])):
                trainingT[i][j] = encode.index(trainingT[i][j])
        # If i represents menopause
        elif i==1:
            # Iterate through all of the columns and add the encode value to the meno list
            for j in range(len(trainingT[i])):
                encode = meno[trainingT[i][j]]
                menoAddCols.append(encode)
        # If i represents breat quadrant
        elif i==7:
            # Iterate through all of the columns and add the encode value to the quad list
            for j in range(len(trainingT[i])):
                encode = quad[trainingT[i][j]]
                quadAddCols.append(encode)

    # Do the same for the test set
    menoAddColsTest = []
    quadAddColsTest = []
    # Iterate through all rows of the test set
    for i in range(len(testingT)):
        encode = []
        # Change all values for features that does not require one-hot encoding 
        if i in features:
            encode = features[i]
            for j in range(len(testingT[i])):
                testingT[i][j] = encode.index(testingT[i][j])
        # Added menopause vals to the column list
        elif i==1:
            for j in range(len(testingT[i])):
                encode = meno[testingT[i][j]]
                menoAddColsTest.append(encode)
        # Added quadrant vals to the column list
        elif i==7:
            for j in range(len(testingT[i])):
                encode = quad[testingT[i][j]]
                quadAddColsTest.append(encode)

    # Untranspose the data back to original format
    trainingInputs = np.transpose(trainingT)
    testingInputs = np.transpose(testingT)
    # Use two counters for each list
    counterMeno=0
    counterQuad=0
    # Iterate through every value in the training input
    for i in range(len(trainingInputs)):
        for j in range(len(trainingInputs[i])):
            # If  we are in the menopause col
            if j==1:
                # Find the correspondinng encode index and add it to finalized training input
                for num in menoAddCols[counterMeno]:
                    finalTrainingInput.append(num)
                # Increment counter
                counterMeno+=1
            # If in quadrant col
            elif j==7:
                # Find corresponding index and append all vals to finalized input
                for num in quadAddCols[counterQuad]:
                    finalTrainingInput.append(num)
                # Increment counter
                counterQuad+=1 
            else:
            # For anything else, just add the value
                finalTrainingInput.append(trainingInputs[i][j])
    # Do the same thing for the test inputs
    counterMeno=0
    counterQuad=0
    for i in range(len(testingInputs)):
        for j in range(len(testingInputs[i])):
            if j==1:
                for num in menoAddColsTest[counterMeno]:
                    finalTestingInput.append(num)
                counterMeno+=1
            elif j==7:
                for num in quadAddColsTest[counterQuad]:
                    finalTestingInput.append(num)
                counterQuad+=1 
            else:
                finalTestingInput.append(testingInputs[i][j])
    
    # Convert to numpy arrays
    finalTrainingInput=np.array(finalTrainingInput)
    finalTestingInput=np.array(finalTestingInput)
    # Reshape them into 2d arrays
    finalTrainingInput=finalTrainingInput.reshape(214,15)
    finalTestingInput=finalTestingInput.reshape(72,15)
    # Convert training data
    for label in training_labels:
        trainingLabel.append(int(label == "recurrence-events"))
    for label in testing_labels:
        testingLabel.append(int(label == "recurrence-events"))
    # ^^^^^ YOUR CODE GOES HERE ^^^^^ $
    return finalTrainingInput, finalTestingInput, trainingLabel, testingLabel



# Hint: consider how to utilize np.argsort()
def k_nearest_neighbors(predict_on, reference_points, reference_labels, k, l, weighted):
    assert len(predict_on) > 0, f"parameter predict_on needs to be of length 0 or greater"
    assert len(reference_points) > 0, f"parameter reference_points needs to be of length 0 or greater"
    assert len(reference_labels) > 0, f"parameter reference_labels needs to be of length 0 or greater"
    assert len(reference_labels) == len(reference_points), f"reference_points and reference_labels need to be the" \
                                                           f" same length"
    predictions = []
    # VVVVV YOUR CODE GOES HERE VVVVV $
    # Iterate through all data points
    for dataPoint in predict_on:
        distances = []
        # Iterate through all neighbors
        for refPoint in reference_points:
            # Append their distances to list
            distances.append((edit_distance(refPoint, dataPoint, l)))
        # Sort reference points by distance
        indices = np.argsort(np.array(distances))
        numRecurrences = 0
        # Iterate through k nearest neighbors and find recurrences
        for i in range(k):
            if reference_labels[indices[i]] == 1:
                numRecurrences += 1
        # If number of recurrences is above a threshold, predict 1
        if numRecurrences * 2 >= k:
            predictions.append(1)
        # Predict 0 if not
        else:
            predictions.append(0)

    # ^^^^^ YOUR CODE GOES HERE ^^^^^ $
    return predictions
