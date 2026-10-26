import pandas as pd # Hier wird die Bibliothek pandas importiert, die für die Datenanalyse und -manipulation in Python verwendet wird.
import re  # Bibliothek für reguläre Ausdrücke
import spacy # Hier wird die Bibliothek spaCy importiert.
nlp = spacy.load("en_core_web_sm")  # Englisches Sprachmodell laden

dateipfad = r"C:\Users\mbbor\OneDrive\Dokumente\IU Universität\Projekt Data Analysis\GitRepository\electronic products.csv" # Pfad zur CSV-Datei

df = pd.read_csv(dateipfad) # DataFrame aus CSV-Datei erstellen df=Dataframe

print(df.head()) # Die ersten fünf Zeilen des DataFrames werden ausgegeben, um einen Überblick über die Daten zu erhalten.

print(df["reviews.text"].head()) # Die ersten fünf Einträge der Spalte "reviews.text" werden ausgegeben, um einen Einblick in die Kundenbewertungen zu erhalten.

print(df["reviews.text"].isna().sum()) # Die Anzahl der fehlenden Werte in der Spalte "reviews.text" wird ausgegeben, um zu sehen, wie viele Bewertungen fehlen.
# Festgestellt, dass es 1 fehlenden Wert in der Spalte "reviews.text" gibt.

df = df.dropna(subset=["reviews.text"]) # Alle Zeilen, die fehlende Werte in der Spalte "reviews.text" enthalten, werden aus dem DataFrame entfernt.
# Diese Zeile wo der Wert fehlt ist entfernt.

reviews = df["reviews.text"] # Die Spalte "reviews.text" wird in eine neue Variable namens "reviews" gespeichert, um die Kundenbewertungen zu analysieren.

print(reviews.iloc[0]) # Die erste Kundenbewertung wird ausgegeben, um einen Einblick in den Inhalt der Bewertungen zu erhalten.

reviews = reviews.str.lower() # Alle Kundenbewertungen werden in Kleinbuchstaben umgewandelt.

print(reviews.iloc[0]) # Prüfen ob alles kleine geschrieben wurde.

def link_entfernen(text): # Diese Funktion entfernt alle Links aus dem gegebenen Text.
    return re.sub(r"https?://\S+|www\.\S+", "", text) # Die Funktion verwendet reguläre Ausdrücke, um Muster zu finden, die mit "http://" oder "https://" beginnen, gefolgt von beliebigen Zeichen, und ersetzt diese durch einen leeren String, wodurch die Links entfernt werden.

reviews = reviews.apply(link_entfernen) # Alle Kundenbewertungen werden durch die Funktion "link_entfernen" verarbeitet, um alle Links aus den Bewertungen zu entfernen.

def satzzeichen_entfernen(text): # Diese Funktion entfernt alle Satzzeichen aus dem gegebenen Text.
    return re.sub(r"[^\w\s]", "", text) # Die Funktion verwendet reguläre Ausdrücke, um alle Zeichen zu finden, die keine Buchstaben, Zahlen oder Leerzeichen sind, und ersetzt diese durch einen leeren String, wodurch die Satzzeichen entfernt werden.

reviews = reviews.apply(satzzeichen_entfernen) # Alle Kundenbewertungen werden durch die Funktion "satzzeichen_entfernen" verarbeitet, um alle Satzzeichen aus den Bewertungen zu entfernen.

print(reviews.iloc[0]) # Die erste Kundenbewertung wird erneut ausgegeben, um zu überprüfen, ob die Links und Satzzeichen erfolgreich entfernt wurden.

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
