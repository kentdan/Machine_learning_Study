#examples
# %% codeblock
#for repeatedly
planets = ['Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune']
for planet in planets:
    print(planet, end=' ') # print all on same line
# %% codeblock
multiplicands = (2, 2, 2, 3, 3, 5)
product = 1
for mult in multiplicands:
    product = product * mult
product
# %% codeblock
s = 'steganograpHy is the practicE of conceaLing a file, message, image, or video within another fiLe, message, image, Or video.'
msg = ''
# print all the uppercase letters in s, one at a time
for char in s:
    if char.isupper():
        print(char, end='')
# %% codeblock
for i in range(5):
    print("Doing important work. i =", i)
# %% codeblock
#while when condition meet
i = 0
while i < 10:
    print(i, end=' ')
    i += 1 # increase the value of i by 1
# %% codeblock
#list comprahension
squares = [n**2 for n in range(10)]
squares
# %% codeblock
#without list comprehension
squares = []
for n in range(10):
    squares.append(n**2)
squares
# %% codeblock
#adding if condition:
short_planets = [planet for planet in planets if len(planet) < 6]
short_planets
# %% codeblock
# str.upper() returns an all-caps version of a string
loud_short_planets =[
    planet.upper() + '!'
    for planet in planets
    if len(planet) < 6
    ]
loud_short_planets
#(Continuing the SQL analogy, you could think of these three lines as SELECT, FROM, and WHERE)
# %% codeblock
def count_negatives(nums):
    """Return the number of negative numbers in the given list.

    >>> count_negatives([5, -1, -2, 0, 3])
    2
    """
    n_negative = 0
    for num in nums:
        if num < 0:
            n_negative = n_negative + 1
    return n_negative
# %% codeblock
#shorter version
# %% codeblock
def count_negatives(nums):
    return len([num for num in nums if num < 0])
# %% codeblock
def count_negatives(nums):
    # Reminder: in the "booleans and conditionals" exercises, we learned about a quirk of
    # Python where it calculates something like True + True + False + True to be equal to 3.
    return sum([num < 0 for num in nums])
#more shorter
# %% codeblock
#Q1
def has_lucky_number(nums):
    """Return whether the given list of numbers is lucky. A lucky list contains
    at least one number divisible by 7.
    """
    return any([num % 7 == 0 for num in nums])
# %% codeblock
#Q2
def elementwise_greater_than(L, thresh):
    """Return a list with the same length as L, where the value at index i is
    True if L[i] is greater than thresh, and False otherwise.

    >>> elementwise_greater_than([1, 2, 3, 4], 2)
    [False, False, True, True]
    """

    return [ele > thresh for ele in L]

# Check your answer
q2.check()
# %% codeblock
#Q3
def menu_is_boring(meals):
    """Given a list of meals served over some period of time, return True if the
    same meal has ever been served two days in a row, and False otherwise.
    """
    for i in range(len(meals)-1):
        if meals[i] == meals[i+1]:
            return True
    return False
​
# Check your answer
q3.check()
# %% codeblock
#Q4

def estimate_average_slot_payout(n_runs):
    result = []
    for a in range(1, n_runs):
        result.append(play_slot_machine())
    return (sum(result) - n_runs)/n_runs

# %% codeblock
