# you can add imports but you should not rely on libraries that are not already provided in "requirements.txt #
from collections import deque

def breadth_first_search(stack):
    # logic: treat each stack orientation as it's own node and perform search until goal orientation reached

    # declare necessary variables
    n = stack.num_books
    q = deque([])
    seen = set()
    q.append((stack, [])) # append stack and list of transformations taken to get to the stack
    seen.add(str(stack)) # used to track orientations we have already seen
    
    # Iterate through the queue
    while q:
        node = q.popleft() # grab the current stack and sequence
        stackCurr = node[0]
        seqCurr = node[1]

        # check if that stack is in the correct order
        if stackCurr.check_ordered():
            # if it is correcty ordered and all 1s, return current sequence
            if 0 not in stackCurr.orientations:
                return seqCurr
        
        # iterate through the stack
        for i in range(1, n+1):
            # make a copy of the current orientation
            stackCurrCopy = stackCurr.copy()
            # flip at i
            stackCurrCopy.flip_stack(i)
            # if this orientation is new, append to our queue with an additional transformation and add to set
            if str(stackCurrCopy) not in seen:
                q.append((stackCurrCopy, seqCurr+[i]) )
                seen.add(str(stackCurrCopy))


def depth_first_search(stack):
    # logic: similar to BFS but with stack
    # declare necessary variables
    n = stack.num_books
    stackDFS = []
    seen = set()
    stackDFS.append((stack, []))
    
    # iterate through stack
    while stackDFS:
        node = stackDFS.pop()
        stackCurr = node[0]
        seqCurr = node[1]

        # skip node if already seen
        if str(stackCurr) in seen:
            continue
        seen.add(str(stackCurr))

        # if stack is ordered and all 1s, return current sequence
        if stackCurr.check_ordered():
            if 0 not in stackCurr.orientations:
                return seqCurr
        
        # iterate through stack
        for i in range(1, n+1):
            # make copy and flip
            stackCurrCopy = stackCurr.copy()
            stackCurrCopy.flip_stack(i)
            # append to stack with transformation
            stackDFS.append((stackCurrCopy, seqCurr+[i]) )