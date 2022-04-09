#examples
# %% codeblock
primes = [2, 3, 5, 7]
planets = ['Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune']
hands = [
    ['J', 'Q', 'K'],
    ['2', '2', '2'],
    ['6', 'A', 'K'], # (Comma after the last element is optional)
]
# (I could also have written this on one line, but it can get hard to read)
hands = [['J', 'Q', 'K'], ['2', '2', '2'], ['6', 'A', 'K']]
my_favourite_things = [32, 'raindrops on roses', help]
# (Yes, Python's help function is *definitely* one of my favourite things)
# %% codeblock
planets[0]
planets[1]
#from right
planets[-1]
planets[-2]
#slicing
planets[0:3]
planets[:3]
planets[3:]
# All the planets except the first and last
planets[1:-1]
# The last 3 planets
planets[-3:]
# %% codeblock
#changing list
planets[3] = 'Malacandra'
planets
planets[:3] = ['Mur', 'Vee', 'Ur']
print(planets)
# That was silly. Let's give them back their old names
planets[:4] = ['Mercury', 'Venus', 'Earth', 'Mars',]
#list.append modifies a list by adding an item to the end:
planets.append('Pluto')
planets
# %% codeblock
# How many planets are there?
len(planets)
# The planets sorted in alphabetical order
sorted(planets)
primes = [2, 3, 5, 7]
sum(primes)
max(primes)
# Interlude: objects
x = 12
# x is a real number, so its imaginary part is 0.
print(x.imag)
# Here's how to make a complex number, in case you've ever been curious:
c = 12 + 3j
print(c.imag)
x.bit_length
x.bit_length()
help(x.bit_length)

##Tuples
# %% codeblock
t = (1, 2, 3)
t = 1, 2, 3 # equivalent to above
t
x = 0.125
x.as_integer_ratio()
numerator, denominator = x.as_integer_ratio()
print(numerator / denominator)
a = 1
b = 0
a, b = b, a
print(a, b)
################練習
#Q1
def select_second(L):
    if len(L) > 2:
        return L[1]
    return None

#Q2
def losing_team_captain(teams):
    return teams[-1][1]
    pass
#Q3
def purple_shell(racers):
    """Given a list of racers, set the first place racer (at the front of the list) to last
    place and vice versa.

    >>> r = ["Mario", "Bowser", "Luigi"]
    >>> purple_shell(r)
    >>> r
    ["Luigi", "Bowser", "Mario"]
    """
    a = racers[0]
    racers[0] = racers[-1]
    racers[-1] = a
    pass

# Check your answer
q3.check()
#Q4
a = [1, 2, 3]
b = [1, [2, 3]]
c = []
d = [1, 2, 3][1:]

# Put your predictions in the list below. Lengths should contain 4 numbers, the
# first being the length of a, the second being the length of b and so on.
lengths = [3,2,0,2]

# Check your answer
q4.check()
#Q5
def fashionably_late(arrivals, name):
    """Given an ordered list of arrivals to the party and a name, return whether the guest with that
    name was fashionably late.
    """
    a = arrivals.index(name)
    return a >= len(arrivals) / 2 and a != len(arrivals) - 1
    pass
​
