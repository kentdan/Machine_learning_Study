#spam
spam_amount = 0
print(spam_amount)

# Ordering Spam, egg, Spam, Spam, bacon and Spam (4 more servings of Spam)
spam_amount = spam_amount + 4

if spam_amount > 0:
    print("But I don't want ANY spam!")

viking_song = "Spam " * spam_amount
print(viking_song)

###Q1
pi = 3.14159 # approximate
diameter = 3

# Create a variable called 'radius' equal to half the diameter
radius = diameter/2

# Create a variable called 'area', using the formula for the area of a circle: pi times the radius squared
area = pi*(radius**2)

# Check your answer
q1.check()

###Q2
########### Setup code - don't touch this part ######################
# If you're curious, these are examples of lists. We'll talk about
# them in depth a few lessons from now. For now, just know that they're
# yet another type of Python object, like int or float.
a = [1, 2, 3]
b = [3, 2, 1]
q2.store_original_ids()
######################################################################
​
c = a
a = b
b = c
# Your code goes here. Swap the values to which a and b refer.
# If you get stuck, you can always uncomment one or both of the lines in
# the next cell for a hint, or to peek at the solution.
​
######################################################################
​###Q3
(5 - 3 )// 2
(5 - 3 )// 2
​###Q4
# Variables representing the number of candies collected by alice, bob, and carol
alice_candies = 121
bob_candies = 77
carol_candies = 109

# involving alice_candies, bob_candies, and carol_candies
to_smash = (alice_candies + bob_candies + carol_candies) % 3
#modulus itu sisanya

# Check your answer
q4.check()
