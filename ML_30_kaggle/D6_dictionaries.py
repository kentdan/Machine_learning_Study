#examples
# %% codeblock
x = 'Pluto is a planet'
y = "Pluto is a planet"
x == y
# %% codeblock
hello = "hello\nworld"
print(hello)
# %% codeblock
triplequoted_hello = """hello
world"""
print(triplequoted_hello)
triplequoted_hello == hello
# %% codeblock
#The print() function automatically adds a newline
print("hello")
print("world")
print("hello", end='')
print("pluto", end='')
# %% codeblock
#Strings can be thought of as sequences of characters
planet = 'Pluto'
planet[0]
# Slicing
planet[-3:]
# How long is this string?
len(planet)
# Yes, we can even loop over them
[char+'! ' for char in planet]
# %% codeblock
# ALL CAPS
claim = "Pluto is a planet!"
claim.upper()
# %% codeblock
claim.startswith(planet)
# %% codeblock
datestr = '1956-01-31'
year, month, day = datestr.split('-')
'/'.join([month, day, year])
# %% codeblock
# Yes, we can put unicode characters right in our string literals :)
words = claim.split()
words
' 👏 '.join([word.upper() for word in words])
# %% codeblock
pluto_mass = 1.303 * 10**22
earth_mass = 5.9722 * 10**24
population = 52910390
#         2 decimal points   3 decimal points, format as percent     separate with commas
"{} weighs about {:.2} kilograms ({:.3%} of Earth's mass). It is home to {:,} Plutonians.".format(
    planet, pluto_mass, pluto_mass / earth_mass, population,
)
# %% codeblock
numbers = {'one':1, 'two':2, 'three':3}
numbers['eleven'] = 11
numbers
numbers['one'] = 'Pluto'
numbers
# %% codeblock
planets = ['Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune']
planet_to_initial = {planet: planet[0] for planet in planets}
planet_to_initial
'Saturn' in planet_to_initial
# %% codeblock
for k in numbers:
    print("{} = {}".format(k, numbers[k]))
# %% codeblock
# Get all the initials, sort them alphabetically, and put them in a space-separated string.
' '.join(sorted(planet_to_initial.values()))
# %% codeblock
for planet, initial in planet_to_initial.items():
    print("{} begins with \"{}\"".format(planet.rjust(10), initial))
# %% codeblock
#Q1
def is_valid_zip(zip_code):
    """Returns whether the input string is a valid (5 digit) zip code
    """
    return len(zip_code) == 5 and zip_code.isdigit()
# %% codeblock
#Q2
def word_search(doc_list, keyword):
    """
    Takes a list of documents (each document is a string) and a keyword.
    Returns list of the index values into the original list for all documents
    containing the keyword.

    Example:
    doc_list = ["The Learn Python Challenge Casino.", "They bought a car", "Casinoville"]
    >>> word_search(doc_list, 'casino')
    >>> [0]
    """
    # list to hold the indices of matching documents
    indices = []
    # Iterate through the indices (i) and elements (doc) of documents
    for i, doc in enumerate(doc_list):
        # Split the string doc into a list of words (according to whitespace)
        tokens = doc.split()
        # Make a transformed list where we 'normalize' each word to facilitate matching.
        # Periods and commas are removed from the end of each word, and it's set to all lowercase.
        normalized = [token.rstrip('.,').lower() for token in tokens]
        # Is there a match? If so, update the list of matching indices.
        if keyword.lower() in normalized:
            indices.append(i)
    return indices
doc_list = ["The Learn Python Challenge Casino.", "They bought a car", "Casinoville"]
word_search(doc_list, 'casino')
# %% codeblock
#Q3
doc_list = ["The Learn Python Challenge Casino.", "They bought a car and a casino", "Casinoville"]
def multi_word_search(doc_list, keywords):
    keyword_to_indices = {}
    for keyword in keywords:
        keyword_to_indices[keyword] = word_search(doc_list, keyword)
    return keyword_to_indices
multi_word_search(doc_list, 'casino')
# %% codeblock
