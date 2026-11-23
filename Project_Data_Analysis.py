import pandas as pd # Hier wird die Bibliothek pandas importiert, die für die Datenanalyse und -manipulation in Python verwendet wird.
import re  # Bibliothek für reguläre Ausdrücke
import spacy # Hier wird die Bibliothek spaCy importiert.
nlp = spacy.load("en_core_web_sm")  # Englisches Sprachmodell laden
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD # Importiert die Singulärwertzerlegung für die LSA.
from sklearn.decomposition import LatentDirichletAllocation # Importiert die Latent Dirichlet Allocation für die LDA.

### CSV-Datei einlesen, Kundenbewertungen prüfen, fehlende Werte entfernen und Bewertungen für die weitere Analyse speichern.
dateipfad = r"C:\Users\mbbor\OneDrive\Dokumente\IU Universität\Projekt Data Analysis\GitRepository\electronic products.csv" # Pfad zur CSV-Datei

df = pd.read_csv(dateipfad) 

print(df.head()) 

print(df["reviews.text"].head()) 

print(df["reviews.text"].isna().sum())

df = df.dropna(subset=["reviews.text"])

reviews = df["reviews.text"]


### Kundenbewertungen zur Vereinheitlichung in Kleinbuchstaben umwandeln und das Ergebnis überprüfen.
print(reviews.iloc[0])

reviews = reviews.str.lower()

print(reviews.iloc[0])


### Funktion zur Entfernung von URLs aus den Kundenbewertungen, um störende Links aus dem Text zu entfernen.
def link_entfernen(text):
    return re.sub(r"https?://\S+|www\.\S+", "", text)

reviews = reviews.apply(link_entfernen)


### Funktion zur Entfernung von Satzzeichen aus den Kundenbewertungen und anschließende Überprüfung der Textbereinigung.
def satzzeichen_entfernen(text):
    return re.sub(r"[^\w\s]", "", text)

reviews = reviews.apply(satzzeichen_entfernen)

print(reviews.iloc[0])


### Fnktion zur Lemmatisierung der Kundenbewertungen, bei der Wörter auf ihre Grundform zurückgeführt und die Ergebnisse überprüft werden.
def lemmatisieren(reviews):
    lemmatisierte_reviews = []

    for bewertung in nlp.pipe(reviews, batch_size=1000):
        woerter = []

        for token in bewertung:
            woerter.append(token.lemma_)

        lemmatisierter_text = " ".join(woerter)

        lemmatisierte_reviews.append(lemmatisierter_text)

    return lemmatisierte_reviews

reviews = lemmatisieren(reviews)

print(reviews[0])


### Erstellung einer Bag-of-Words-Matrix zur Darstellung der Worthäufigkeiten. Dabei werden englische Stoppwörter entfernt, außer „not“ und „no“, um wichtige Negationen zu erhalten.
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

stop_words = ENGLISH_STOP_WORDS - {"not", "no"}

vectorizer = CountVectorizer(stop_words=list(stop_words))

bow_matrix = vectorizer.fit_transform(reviews)

vocabulary = vectorizer.get_feature_names_out()


### Ausgabe der Größe des Vokabulars und der Dimensionen der BoW-Matrix zur Überprüfung der Ergebnisse.
print("Anzahl der Wörter im Vokabular:", len(vocabulary))

print("Dimensionen der BoW-Matrix:", bow_matrix.shape)


### Ausgabe der Worthäufigkeiten der ersten Kundenbewertung zur Überprüfung der BoW-Darstellung.
wort_haeufigkeiten = bow_matrix[0].toarray()[0]
for index, anzahl in enumerate(wort_haeufigkeiten):
    if anzahl > 0:
        print(vocabulary[index], anzahl)


### Ermittlung und Ausgabe der zehn häufigsten Wörter über alle Kundenbewertungen hinweg.
wort_gesamt = bow_matrix.sum(axis=0).A1

haeufigste_indizes = wort_gesamt.argsort()[-10:][::-1]

for index in haeufigste_indizes:
    print(vocabulary[index], wort_gesamt[index])


### Erstellung einer TF-IDF-Matrix zur Gewichtung der Wörter nach ihrer Relevanz in den Kundenbewertungen und Ausgabe der Matrixdimensionen.
tfidf_vectorizer = TfidfVectorizer(min_df=1, stop_words=list(stop_words))

tfidf_matrix = tfidf_vectorizer.fit_transform(reviews)

tfidf_vocabulary = tfidf_vectorizer.get_feature_names_out()

print("Dimensionen der TF-IDF-Matrix:", tfidf_matrix.shape)


### Ausgabe der zehn Wörter mit den höchsten TF-IDF-Werten der ersten Kundenbewertung.
tfidf_werte = tfidf_matrix[0].toarray()[0]
sortierte_indizes = tfidf_werte.argsort()[::-1]

for index in sortierte_indizes[:10]:
    if tfidf_werte[index] > 0:
        print(tfidf_vocabulary[index], round(tfidf_werte[index], 3))


### Gegenüberstellung der BoW-Häufigkeiten und TF-IDF-Gewichtungen aller Wörter der ersten Kundenbewertung.
vergleich = pd.DataFrame({
    "Wort": vocabulary,
    "BoW": bow_matrix[0].toarray()[0],
    "TF-IDF": tfidf_matrix[0].toarray()[0]
})

vergleich = vergleich[vergleich["BoW"] > 0]

vergleich = vergleich.sort_values(by="TF-IDF", ascending=False)

print(vergleich.to_string(index=False))


### Durchführung der Latent Semantic Analysis (LSA) zur Reduzierung der TF-IDF-Matrix auf 20 Komponenten. Anschließend werden die wichtigsten Begriffe, die erklärte Varianz und die Verteilung der Bewertungen auf die Komponenten untersucht.
LSA_model = TruncatedSVD(n_components=20, algorithm='randomized', n_iter=10, random_state=42) # LSA-Modell mit 20 Komponenten erstellen.

lsa_matrix = LSA_model.fit_transform(tfidf_matrix)

print("Dimensionen der LSA-Matrix:", lsa_matrix.shape)

print("Gesamte erklärte Varianz:", round(LSA_model.explained_variance_ratio_.sum() * 100, 2), "%")


begriffe = tfidf_vectorizer.get_feature_names_out()

for i, komponente in enumerate(LSA_model.components_):
    top_woerter = komponente.argsort()[-10:][::-1]

    print(f"\nThema {i + 1}:")
    print(", ".join(begriffe[top_woerter]))

haeufigste_komponenten = lsa_matrix.argmax(axis=1)

komponenten_anzahl = pd.Series(haeufigste_komponenten).value_counts().reindex(range(20), fill_value=0).sort_index() 

for komponente, anzahl in komponenten_anzahl.items():
    prozent = anzahl / len(reviews) * 100
    print(f"Thema {komponente + 1}: {anzahl} Bewertungen ({prozent:.2f} %)")


### Durchführung der Latent Dirichlet Allocation (LDA) zur Identifikation von 20 Themen auf Basis der BoW-Matrix. Anschließend werden die wichtigsten Begriffe und die Verteilung der Bewertungen auf die Themen untersucht.
LDA_model = LatentDirichletAllocation(n_components=20, random_state=42)
lda_matrix = LDA_model.fit_transform(bow_matrix)
print("Dimensionen der LDA-Matrix:", lda_matrix.shape)


begriffe = vectorizer.get_feature_names_out()

for i, thema in enumerate(LDA_model.components_):
    top_woerter = thema.argsort()[-10:][::-1]

    print(f"\nThema {i + 1}:")
    print(", ".join(begriffe[top_woerter])) 

haeufigste_themen = lda_matrix.argmax(axis=1)

themen_anzahl = pd.Series(haeufigste_themen).value_counts().reindex(range(20), fill_value=0).sort_index()

for thema, anzahl in themen_anzahl.items():
    prozent = anzahl / len(reviews) * 100 
    print(f"Thema {thema + 1}: {anzahl} Bewertungen ({prozent:.2f} %)")