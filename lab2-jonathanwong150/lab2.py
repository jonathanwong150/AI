# you can add imports but you should not rely on libraries that are not already provided in "requirements.txt #
from heapq import heappush, heappop
from lab2_utils import TextbookStack, apply_sequence
from collections import deque


def a_star_search(stack):

    flip_sequence = []

    # --- v ADD YOUR CODE HERE v --- #
    def heuristic(stack):
        n=stack.num_books
        h=0
        # Iterate through bookstack
        for i in range(1,n):
            # Store the current and previous book numbers and face
            currBook, currBookFace=stack.order[i], stack.orientations[i]
            prevBook, prevBookFace=stack.order[i-1], stack.orientations[i-1]
            # Increment heuristic if it follows a certain set of criteria
            if abs(currBook-prevBook) > 1 or currBookFace != prevBookFace or (prevBook > currBook and prevBookFace) or prevBookFace == 0:
                h += 1 
        return h

    # Create priority queue that will store tuples with values needed for A*; allows us to heappop node with lowest heuristic
    n=stack.num_books
    priorityQueue=[]
    heappush(priorityQueue,(0,0,0,[],stack))
    seen=set()
    # Iterate through the priority queue
    while priorityQueue:
        # pop and store the values of our current stack and sequence from most desireable heap node
        node=heappop(priorityQueue)
        currSeq=node[3]
        currStack=node[4]
        # If stack is new, check if it is in order
        if str(currStack) not in seen:
            if currStack.check_ordered():
                return currSeq
            # If current stack is not in order, add to our set and manipulate it
            seen.add(str(currStack))
            # Iterate through bookstack
            for i in range(1,n+1):
                # Ensure we do not waste moves by re-flipping the last flipped book
                if currSeq:
                    lastMove=currSeq[-1]
                else:
                    lastMove=0
                if i!=lastMove:
                    # Make a copy of the current stack at each index and flip at i
                    nextStack=currStack.copy()
                    nextStack.flip_stack(i)
                    # Calculate the heuristic and g values of the new stack
                    h=heuristic(nextStack)
                    g=node[2]+1
                    f=h+g
                    # Push the new stack into the priority queue
                    heappush(priorityQueue,(f,h,g,currSeq+[i],nextStack))

    # ---------------------------- #


def weighted_a_star_search(stack, epsilon=None, N=1):
    # Weighted A* is extra credit

    flip_sequence = []

    # --- v ADD YOUR CODE HERE v --- #

    return flip_sequence

    # ---------------------------- #