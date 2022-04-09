# %% codeblock
###"""Return the given number rounded to two decimal places.

def round_to_two_places(num):
    return round(num, 2)
round_to_two_places(3.14159)

# %% codeblock
#The help for round says that ndigits (the second argument) may be negative.
#What do you think will happen when it is? Try some examples in the following cell.

round(338.4324123,ndigits=1)

# %% codeblock
def to_smash(total_candies, n_friends=3):
    return total_candies % n_friends
to_smash(46)
# %% codeblock
def ruound_to_two_places(num):
    return round(num,ndigits=3)
ruound_to_two_places(9.9999)
# %% codeblock
x = -10
y = 5
min(x, y)
# %% codeblock
def f(x):
 y = abs(x)
 return y

print(f(5))
# %% codeblock
