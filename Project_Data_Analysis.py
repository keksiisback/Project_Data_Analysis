import pandas as pd # Hier wird die Bibliothek pandas importiert, die für die Datenanalyse und -manipulation in Python verwendet wird.
import re  # Bibliothek für reguläre Ausdrücke
import spacy # Hier wird die Bibliothek spaCy importiert.
nlp = spacy.load("en_core_web_sm")  # Englisches Sprachmodell laden
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


### Daten einlesen
dateipfad = r"C:\Users\mbbor\OneDrive\Dokumente\IU Universität\Projekt Data Analysis\GitRepository\electronic products.csv" # Pfad zur CSV-Datei

df = pd.read_csv(dateipfad) # DataFrame aus CSV-Datei erstellen df=Dataframe

print(df.head()) # Die ersten fünf Zeilen des DataFrames werden ausgegeben, um einen Überblick über die Daten zu erhalten.

print(df["reviews.text"].head()) # Die ersten fünf Einträge der Spalte "reviews.text" werden ausgegeben, um einen Einblick in die Kundenbewertungen zu erhalten.

print(df["reviews.text"].isna().sum()) # Die Anzahl der fehlenden Werte in der Spalte "reviews.text" wird ausgegeben, um zu sehen, wie viele Bewertungen fehlen.
# Festgestellt, dass es 1 fehlenden Wert in der Spalte "reviews.text" gibt.

df = df.dropna(subset=["reviews.text"]) # Alle Zeilen, die fehlende Werte in der Spalte "reviews.text" enthalten, werden aus dem DataFrame entfernt.
# Diese Zeile wo der Wert fehlt ist entfernt.

reviews = df["reviews.text"] # Die Spalte "reviews.text" wird in eine neue Variable namens "reviews" gespeichert, um die Kundenbewertungen zu analysieren.


### Vorverarbeitung der Kundenbewertungen
print(reviews.iloc[0]) # Die erste Kundenbewertung wird ausgegeben, um einen Einblick in den Inhalt der Bewertungen zu erhalten.

reviews = reviews.str.lower() # Alle Kundenbewertungen werden in Kleinbuchstaben umgewandelt.

print(reviews.iloc[0]) # Prüfen ob alles kleine geschrieben wurde.


### Links entfernen
def link_entfernen(text): # Diese Funktion entfernt alle Links aus dem gegebenen Text.
    return re.sub(r"https?://\S+|www\.\S+", "", text) # Die Funktion verwendet reguläre Ausdrücke, um Muster zu finden, die mit "http://" oder "https://" beginnen, gefolgt von beliebigen Zeichen, und ersetzt diese durch einen leeren String, wodurch die Links entfernt werden.

reviews = reviews.apply(link_entfernen) # Alle Kundenbewertungen werden durch die Funktion "link_entfernen" verarbeitet, um alle Links aus den Bewertungen zu entfernen.


### Satzzeichen entfernen
def satzzeichen_entfernen(text): # Diese Funktion entfernt alle Satzzeichen aus dem gegebenen Text.
    return re.sub(r"[^\w\s]", "", text) # Die Funktion verwendet reguläre Ausdrücke, um alle Zeichen zu finden, die keine Buchstaben, Zahlen oder Leerzeichen sind, und ersetzt diese durch einen leeren String, wodurch die Satzzeichen entfernt werden.

reviews = reviews.apply(satzzeichen_entfernen) # Alle Kundenbewertungen werden durch die Funktion "satzzeichen_entfernen" verarbeitet, um alle Satzzeichen aus den Bewertungen zu entfernen.

print(reviews.iloc[0]) # Die erste Kundenbewertung wird erneut ausgegeben, um zu überprüfen, ob die Links und Satzzeichen erfolgreich entfernt wurden.


### Lemmatisierung der Kundenbewertungen
def lemmatisieren(reviews): # Diese Funktion lemmatisiert alle Kundenbewertungen, indem sie die Grundformen der Wörter extrahiert.
    lemmatisierte_reviews = []  # Liste für die lemmatisierten Bewertungen erstellen

    for bewertung in nlp.pipe(reviews, batch_size=1000): # Die Bewertungen werden in Batches verarbeitet, um die Verarbeitung zu optimieren - diese Funktion nimmt etwas Zeit in Anspruch.
        woerter = []  # Liste für die Wörter in der aktuellen Bewertung erstellen

        for token in bewertung: # Jedes Token (Wort) in der aktuellen Bewertung wird durchlaufen
            woerter.append(token.lemma_)  # Das Lemmatisieren wird durchgeführt, indem die Grundform jedes Tokens extrahiert und in der Liste "woerter" gespeichert wird.

        lemmatisierter_text = " ".join(woerter)  # Die lemmatisierten Wörter werden zu einem Text zusammengefügt, indem sie durch Leerzeichen getrennt werden.

        lemmatisierte_reviews.append(lemmatisierter_text)  # Die lemmatisierte Bewertung wird der Liste "lemmatisierte_reviews" hinzugefügt.

    return lemmatisierte_reviews # Die Funktion gibt die Liste der lemmatisierten Bewertungen zurück.

reviews = lemmatisieren(reviews)  # Die Funktion "lemmatisieren" wird auf die Kundenbewertungen angewendet, um die Grundformen der Wörter in den Bewertungen zu extrahieren.

print(reviews[0])  # Die erste lemmatisierte Kundenbewertung wird ausgegeben, um zu überprüfen, ob die Lemmatisierung erfolgreich durchgeführt wurde.


### Erstellen von BoW 
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS # Die Liste der englischen Stoppwörter wird aus der Bibliothek sklearn importiert, um häufig vorkommende Wörter zu identifizieren, die in der Textanalyse möglicherweise nicht relevant sind.

stop_words = ENGLISH_STOP_WORDS - {"not", "no"} # Die Stoppwörter "not" und "no" werden aus der Liste der englischen Stoppwörter entfernt, da sie in der Sentiment-Analyse wichtig sein können, um die Bedeutung von negativen Bewertungen zu erfassen.

vectorizer = CountVectorizer(stop_words=list(stop_words)) # Ein CountVectorizer-Objekt wird erstellt, das die Stoppwörter verwendet, um die Kundenbewertungen in eine Bag-of-Words-Darstellung umzuwandeln. Die Stoppwörter werden aus der Liste der englischen Stoppwörter entfernt, um die Bedeutung von negativen Bewertungen zu erfassen.

bow_matrix = vectorizer.fit_transform(reviews) # Die Kundenbewertungen werden in eine Wort-Häufigkeitsmatrix umgewandelt.

vocabulary = vectorizer.get_feature_names_out() # Das Vokabular (die einzigartigen Wörter) wird aus dem CountVectorizer-Objekt extrahiert.


### Ausgaben von Vokabular und BoW-Matrix
print("Anzahl der Wörter im Vokabular:", len(vocabulary)) # Die Anzahl der einzigartigen Wörter im Vokabular wird ausgegeben.

print("Dimensionen der BoW-Matrix:", bow_matrix.shape) # Die Dimensionen der BoW-Matrix werden ausgegeben.


### Ausgabe der Wort-Häufigkeiten der ersten Bewertung
wort_haeufigkeiten = bow_matrix[0].toarray()[0] # Die Wort-Häufigkeiten der ersten Bewertung werden in ein Array umgewandelt, um die Häufigkeit jedes Wortes in der ersten Bewertung zu analysieren.
for index, anzahl in enumerate(wort_haeufigkeiten): # Die Häufigkeit jedes Wortes in der ersten Bewertung wird durchlaufen, wobei der Index des Wortes und die Häufigkeit des Wortes in der ersten Bewertung zurückgegeben werden.
    if anzahl > 0: # wenn die Häufigkeit eines Wortes größer als 0 ist, wird das Wort und seine Häufigkeit ausgegeben.
        print(vocabulary[index], anzahl) # Das Wort und seine Häufigkeit werden ausgegeben.


### Gesamthäufigkeit der Wörter in allen Bewertungen
wort_gesamt = bow_matrix.sum(axis=0).A1 # Die Gesamthäufigkeit jedes Wortes in allen Bewertungen wird berechnet.

haeufigste_indizes = wort_gesamt.argsort()[-10:][::-1] # Die Indizes der 10 häufigsten Wörter werden ermittelt, indem die Gesamthäufigkeiten der Wörter sortiert und die letzten 10 Indizes ausgewählt werden. Diese Indizes werden dann in umgekehrter Reihenfolge zurückgegeben, um die häufigsten Wörter zuerst anzuzeigen.

for index in haeufigste_indizes: # Die 10 häufigsten Wörter werden durchlaufen, wobei der Index des Wortes zurückgegeben wird.
    print(vocabulary[index], wort_gesamt[index]) # Das häufigste Wort und seine Gesamthäufigkeit werden ausgegeben.


### Ab jetzt erstellen wir die TF-IDF-Matrix.
tfidf_vectorizer = TfidfVectorizer(min_df=1, stop_words=list(stop_words)) # Ein TfidfVectorizer-Objekt wird erstellt, das die Stoppwörter verwendet, um die Kundenbewertungen in eine TF-IDF-Darstellung umzuwandeln. Die Stoppwörter werden aus der Liste der englischen Stoppwörter entfernt, um die Bedeutung von negativen Bewertungen zu erfassen.

tfidf_matrix = tfidf_vectorizer.fit_transform(reviews) # Die Kundenbewertungen werden in eine TF-IDF-Matrix umgewandelt, die die Wichtigkeit jedes Wortes in den Bewertungen basierend auf seiner Häufigkeit und der Anzahl der Bewertungen, in denen es vorkommt, darstellt.

tfidf_vocabulary = tfidf_vectorizer.get_feature_names_out() # Das Vokabular (die einzigartigen Wörter) wird aus dem TfidfVectorizer-Objekt extrahiert.

print("Dimensionen der TF-IDF-Matrix:", tfidf_matrix.shape) # Die Dimensionen der TF-IDF-Matrix werden ausgegeben, um die Anzahl der Bewertungen und die Anzahl der einzigartigen Wörter in der TF-IDF-Darstellung zu überprüfen.


### Ausgabe der TF-IDF-Werte der ersten Bewertung
tfidf_werte = tfidf_matrix[0].toarray()[0] # Die TF-IDF-Werte der ersten Bewertung werden in ein Array umgewandelt.
sortierte_indizes = tfidf_werte.argsort()[::-1] # Die Indizes der Wörter werden nach ihren TF-IDF-Werten in absteigender Reihenfolge sortiert, um die Wörter mit den höchsten TF-IDF-Werten zuerst anzuzeigen.

for index in sortierte_indizes[:10]: # Die 10 Wörter mit den höchsten TF-IDF-Werten werden durchlaufen, wobei der Index des Wortes zurückgegeben wird.
    if tfidf_werte[index] > 0: # wenn der TF-IDF-Wert eines Wortes größer als 0 ist, wird das Wort und sein TF-IDF-Wert ausgegeben.
        print(tfidf_vocabulary[index], round(tfidf_werte[index], 3)) # Das Wort und sein TF-IDF-Wert werden ausgegeben, wobei der TF-IDF-Wert auf drei Dezimalstellen gerundet wird.


### BoW- und TF-IDF-Werte der ersten Bewertung vergleichen
vergleich = pd.DataFrame({
    "Wort": vocabulary,
    "BoW": bow_matrix[0].toarray()[0],
    "TF-IDF": tfidf_matrix[0].toarray()[0]
}) # Ein DataFrame namens "vergleich" wird erstellt, der die Wörter, ihre Häufigkeiten in der ersten Bewertung (BoW) und ihre TF-IDF-Werte enthält.

vergleich = vergleich[vergleich["BoW"] > 0] # Nur Wörter anzeigen, die in der ersten Bewertung vorkommen

vergleich = vergleich.sort_values(by="TF-IDF", ascending=False) # Die Wörter werden nach ihren TF-IDF-Werten in absteigender Reihenfolge sortiert, um die Wörter mit den höchsten TF-IDF-Werten zuerst anzuzeigen.

print(vergleich.to_string(index=False)) # Der DataFrame "vergleich" wird ausgegeben, wobei die Indexspalte nicht angezeigt wird, um die Wörter, ihre Häufigkeiten in der ersten Bewertung (BoW) und ihre TF-IDF-Werte zu vergleichen.
