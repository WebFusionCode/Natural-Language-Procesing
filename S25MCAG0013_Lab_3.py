import nltk
from nltk.corpus import brown
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
from nltk.probability import FreqDist
from wordcloud import WordCloud
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import pandas as pd

# Download required datasets
nltk.download('brown')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Load text from Brown Corpus
category = 'adventure'
words_list = brown.words(categories=category)
text = ' '.join(words_list)
tokens = word_tokenize(text)

print("First 20 tokens:", tokens[:20])
print("Total tokens:", len(tokens))


# Q2: Visualize word frequencies
def visualize_word_frequencies(tokens):
    # Calculate frequency distribution
    fdist = FreqDist(tokens)
    # Show top 20 words as a bar chart
    top_words = fdist.most_common(20)
    words, counts = zip(*top_words)
    plt.figure(figsize=(12,6))
    plt.bar(words, counts)
    plt.title("Top 20 Most Common Words")
    plt.xticks(rotation=45)
    plt.xlabel("Words")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()
    # Generate word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(tokens))
    plt.figure(figsize=(12,6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title("Word Cloud of Tokens")
    plt.show()


# Q3: Apply stemming
def apply_stemming(tokens):
    stemmer = PorterStemmer()
    stems = [stemmer.stem(token) for token in tokens]
    stem_fdist = FreqDist(stems)
    print("Top 20 Stems:")
    for stem, freq in stem_fdist.most_common(20):
        print(f"{stem}: {freq}")
    return stem_fdist


# Q4: Apply lemmatization
def apply_lemmatization(tokens):
    lemmatizer = WordNetLemmatizer()
    lemmas = [lemmatizer.lemmatize(token) for token in tokens]
    lemma_fdist = FreqDist(lemmas)
    print("Top 20 Lemmas:")
    for lemma, freq in lemma_fdist.most_common(20):
        print(f"{lemma}: {freq}")
    return lemma_fdist


# Q5: Compare stemming and lemmatization
def compare_stemming_lemmatization(tokens):
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

    fdist = FreqDist([t for t in tokens if t.isalpha()])
    top_words = [w for w, _ in fdist.most_common(15)]
    data = {
        'Original': [],
        'Stemmed': [],
        'Lemmatized': [],
    }
    for word in top_words:
        data['Original'].append(word)
        data['Stemmed'].append(stemmer.stem(word))
        data['Lemmatized'].append(lemmatizer.lemmatize(word))
    df = pd.DataFrame(data)
    print("\nComparison of Original, Stemmed, and Lemmatized Forms:")
    print(df)
    # Optional: plot a bar chart of counts for each form
    plt.figure(figsize=(12,6))
    x = range(len(df))
    plt.bar(x, [fdist[word] for word in df['Original']], width=0.25, label='Original')
    plt.bar([i+0.25 for i in x], [fdist[data['Stemmed'][i]] if data['Stemmed'][i] in fdist else 0 for i in x], width=0.25, label='Stemmed')
    plt.bar([i+0.5 for i in x], [fdist[data['Lemmatized'][i]] if data['Lemmatized'][i] in fdist else 0 for i in x], width=0.25, label='Lemmatized')
    plt.xticks([i+0.25 for i in x], df['Original'], rotation=45)
    plt.xlabel("Word")
    plt.ylabel("Frequency")
    plt.title("Frequency Comparison: Original vs Stemmed vs Lemmatized")
    plt.legend()
    plt.tight_layout()
    plt.show()


# Call functions in sequence
# visualize_word_frequencies(tokens)
apply_stemming(tokens)
# apply_lemmatization(tokens)
# compare_stemming_lemmatization(tokens)