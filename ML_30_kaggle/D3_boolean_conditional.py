#examples
# %% codeblock
x = True
print(x)
print(type(x))
# %% codeblock
def can_run_for_president(age):
    """Can someone of the given age run for president in the US?"""
    # The US Constitution says you must be at least 35 years old
    return age >= 35

print("Can a 19-year-old run for president?", can_run_for_president(19))
print("Can a 45-year-old run for president?", can_run_for_president(45))
# %% codeblock
3.0 == 3
'3' == 3
# %% codeblock
def is_odd(n):
    return (n % 2) == 1

print("Is 100 odd?", is_odd(100))
print("Is -1 odd?", is_odd(-1))
# %% codeblock
def inspect(x):
    if x == 0:
        print(x, "is zero")
    elif x > 0:
        print(x, "is positive")
    elif x < 0:
        print(x, "is negative")
    else:
        print(x, "is unlike anything I've ever seen...")

inspect(0)
inspect(-15)
# %% codeblock
if 0:
    print(0)
elif "spam":
    print("spam")

# %% codeblock
#Q1
# Your code goes here. Define a function called 'sign'
def sign(x):
    if x < 0:
        return -1
    elif x> 0:
        return 1
    elif x == 0:
       return 0

sign(-25)
# %% codeblock
#Q2
def to_smash(total_candies):
    print("Splitting", total_candies, "candy" if total_candies == 1 else "candies")
    return total_candies % 3

to_smash(91)
to_smash(1)
# %% codeblock
#Q3
def prepared_for_weather(have_umbrella, rain_level, have_hood, is_workday):
    # Don't change this code. Our goal is just to find the bug, not fix it!
    return have_umbrella or (rain_level < 5 and have_hood) or not (rain_level > 0 and is_workday)

# Change the values of these inputs so they represent a case where prepared_for_weather
# returns the wrong answer.
have_umbrella = False
rain_level = 0.0
have_hood = False
is_workday = False

# Check what the function returns given the current values of the variables above
actual = prepared_for_weather(have_umbrella, rain_level, have_hood, is_workday)
print(actual)

q3.check()
# %% codeblock
#Q4
def is_negative(number):
    if number < 0:
        return True
    else:
        return False

def concise_is_negative(number):
    return number <0
concise_is_negative(-40)
# %% codeblock
#Q5a
def wants_all_toppings(ketchup, mustard, onion):
    """Return whether the customer wants "the works" (all 3 toppings)
    """
    return ketchup and mustard and onion
# %% codeblock
#Q5b
def wants_plain_hotdog(ketchup, mustard, onion):
    """Return whether the customer wants a plain hot dog with no toppings.
    """
    pass
    return not (ketchup or mustard or onion)
# %% codeblock
#Q5c
def exactly_one_sauce(ketchup, mustard, onion):
    """Return whether the customer wants either ketchup or mustard, but not both.
    (You may be familiar with this operation under the name "exclusive or")
    """
    pass
    return (ketchup and not mustard) or (mustard and not ketchup)
# %% codeblock
#Q6
def exactly_one_topping(ketchup, mustard, onion):
    """Return whether the customer wants exactly one of the three available toppings
    on their hot dog.
    """
    pass
    return (ketchup + mustard + onion) == 1
# %% codeblock
