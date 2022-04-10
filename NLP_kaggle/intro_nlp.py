# %% codelock
import spacy
nlp = spacy.load('en_core_web_sm')
doc = nlp("Tea is healthy and calming, don't you think?")
# %% codelock
#tokenizing
for token in doc:
    print(token)
# %% codelock
#text preprocessing
print(f"Token \t\tLemma \t\tStopword".format('Token', 'Lemma', 'Stopword'))
print("-"*40)
for token in doc:
    print(f"{str(token)}\t\t{token.lemma_}\t\t{token.is_stop}")
# %% codelock
#pattern maching
from spacy.matcher import PhraseMatcher
matcher = PhraseMatcher(nlp.vocab, attr='LOWER')
# %% codelock
terms = ['Galaxy Note', 'iPhone 11', 'iPhone XS', 'Google Pixel']
patterns = [nlp(text) for text in terms]
matcher.add("TerminologyList", patterns)
# %% codelock
# Borrowed from https://daringfireball.net/linked/2019/09/21/patel-11-pro
text_doc = nlp("Glowing review overall, and some really interesting side-by-side "
               "photography tests pitting the iPhone 11 Pro against the "
               "Galaxy Note 10 Plus and last year’s iPhone XS and Google Pixel 3.")
matches = matcher(text_doc)
print(matches)
# %% codelock
match_id, start, end = matches[0]
print(nlp.vocab.strings[match_id], text_doc[start:end])
# %% codelock
# excercise
import pandas as pd
# Load in the data from JSON file
data = pd.read_json('/Users/danielkent/Documents/Code/Dataset/nlp_course/restaurant.json')
data.head()
# %% codelock
menu = ["Cheese Steak", "Cheesesteak", "Steak and Cheese", "Italian Combo", "Tiramisu", "Cannoli",
        "Chicken Salad", "Chicken Spinach Salad", "Meatball", "Pizza", "Pizzas", "Spaghetti",
        "Bruchetta", "Eggplant", "Italian Beef", "Purista", "Pasta", "Calzones",  "Calzone",
        "Italian Sausage", "Chicken Cutlet", "Chicken Parm", "Chicken Parmesan", "Gnocchi",
        "Chicken Pesto", "Turkey Sandwich", "Turkey Breast", "Ziti", "Portobello", "Reuben",
        "Mozzarella Caprese",  "Corned Beef", "Garlic Bread", "Pastrami", "Roast Beef",
        "Tuna Salad", "Lasagna", "Artichoke Salad", "Fettuccini Alfredo", "Chicken Parmigiana",
        "Grilled Veggie", "Grilled Veggies", "Grilled Vegetable", "Mac and Cheese", "Macaroni",
         "Prosciutto", "Salami"]
# %% codelock
#You'll pursue this plan of calculating average scores of the reviews mentioning each menu item.
#As a first step, you'll write code to extract the foods mentioned in a single review.
#Since menu items are multiple tokens long, you'll use PhraseMatcher which can match series of tokens.
#Fill in the ____ values below to get a list of items matching a single menu item.
import spacy
from spacy.matcher import PhraseMatcher
​# %% codelock
index_of_review_to_test_on = 14
text_to_test_on = data.text.iloc[index_of_review_to_test_on]
​# %% codelock
# Load the SpaCy model
nlp = spacy.blank('en')
​# %% codelock
# Create the tokenized version of text_to_test_on
review_doc = nlp(text_to_test_on)
​# %% codelock
# Create the PhraseMatcher object. The tokenizer is the first argument. Use attr = 'LOWER' to make consistent capitalization
matcher = PhraseMatcher(nlp.vocab, attr='LOWER')
​# %% codelock
# Create a list of tokens for each item in the menu
menu_tokens_list = [nlp(item) for item in menu]
# Add the item patterns to the matcher.
matcher.add("MENU", menu_tokens_list)
matches = matcher(review_doc)
# %% codelock
for match in matches:
   print(f"Token number {match[1]}: {review_doc[match[1]:match[2]]}")
# %% codelock
#step 3 Matching on the whole dataset¶
#maching the whole Dataset
from collections import defaultdict

# item_ratings is a dictionary of lists. If a key doesn't exist in item_ratings,
# the key is added with an empty list as the value.
item_ratings = defaultdict(list)

for idx, review in data.iterrows():
    doc = nlp(review['text'])
    # Using the matcher from the previous exercise
    matches = matcher(doc)

    # Create a set of the items found in the review text
    found_items = {doc[start_idx:end_idx] for (_, start_idx, end_idx) in matches}

    # Update item_ratings with rating for each item in found_items
    # Transform the item strings to lowercase to make it case insensitive
    for item in found_items:
        item_ratings[item.text.lower()].append(review['stars'])
# %% codelock
print(found_items)
#worst # REVIEW:
mean_ratings = {item: sum(ratings)/len(ratings) for item, ratings in item_ratings.items()}
worst_item = sorted(mean_ratings, key=mean_ratings.get)[0]
# %% codelock
print(worst_item)
print(mean_ratings[worst_item])
# %% codelock
#count # REVIEW:
counts = {item: len(ratings) for item, ratings in item_ratings.items()}

item_counts = sorted(counts, key=counts.get, reverse=True)
for item in item_counts:
    print(f"{item:>25}{counts[item]:>5}")
# %% codelock
#sorting
sorted_ratings = sorted(mean_ratings, key=mean_ratings.get)
# %% codelock
print("Worst rated menu items:")
for item in sorted_ratings[:10]:
    print(f"{item:20} Ave rating: {mean_ratings[item]:.2f} \tcount: {counts[item]}")
# %% codelock
print("\n\nBest rated menu items:")
for item in sorted_ratings[-10:]:
    print(f"{item:20} Ave rating: {mean_ratings[item]:.2f} \tcount: {counts[item]}")
#The less data you have for any specific item,
#the less you can trust that the average rating is the "real" sentiment of the customers.
