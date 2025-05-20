## Jakub Woś
## 288581
## informatyka ogolnoakademicka

# Dokumentacja Projektu: Analiza Cen Nieruchomości i Detekcja Fałszywych Wiadomości
## 1. Cel Projektu

Projekt składa się z dwóch niezależnych modułów, których celem jest demonstracja zastosowania metod uczenia maszynowego w dwóch różnych dziedzinach:

    House Prices Prediction – przewidywanie cen domów w Bostonie przy użyciu modeli regresji.

    Fake News Detection – klasyfikacja tekstów na prawdziwe i fałszywe za pomocą sieci neuronowych.

Oba programy wykorzystują rzeczywiste zbiory danych oraz zaawansowane techniki przetwarzania i modelowania, aby osiągnąć swoje cele.
## 2. Opis Projektów
### 2.1 House Prices Prediction
Cel

Przewidywanie mediany cen domów (MEDV) w Bostonie na podstawie wybranych cech, takich jak liczba pokoi (RM), odległość od centrów zatrudnienia (DIS), czy wskaźnik przestępczości (CRIM).
Dane

    Źródło: Zmodyfikowana wersja zbioru Boston Housing.

    Cechy:

        Numeryczne: CRIM, RM, AGE, DIS.

        Kategoryczna: CHAS (bliskość do rzeki).

    Target: MEDV (mediana wartości domów w tysiącach dolarów).

Działanie Programu

    Przetwarzanie danych:

        Standaryzacja cech numerycznych (StandardScaler).

        Kodowanie kategorycznej cechy CHAS (OneHotEncoder).

    Podział danych: 80% treningowe, 20% testowe.

    Modele:

        Regresja liniowa.

        Drzewo decyzyjne (max głębokość: 5).

        Random Forest (100 estymatorów).

        XGBoost (100 estymatorów, funkcja straty MSE).

    Ewaluacja:

        Metryki: MSE (błąd średniokwadratowy) i R² (wskaźnik determinacji).

        Wizualizacja: Wykresy rzeczywistych vs. przewidywanych wartości.

Wyniki

Program porównuje wydajność modeli i wskazuje najlepszy na podstawie R². Przykładowy wynik:

Najlepszy model: Random Forest (R² = 0.872)

### 2.2 Fake News Detection
Cel

Klasyfikacja wypowiedzi politycznych na prawdziwe (half-true, mostly-true, true) i fałszywe (pants-fire, false, barely-true).
Dane

    Źródło: Zbiór LIAR.

    Struktura:

        Treningowy: train.tsv.

        Testowy: test.tsv.

        Walidacyjny: valid.tsv.

    Cechy tekstowe: statement (treść wypowiedzi).

    Target: Etykieta binarna (0 = fałsz, 1 = prawda).

Działanie Programu

    Przetwarzanie tekstu:

        Tokenizacja i padding sekwencji (Tokenizer, pad_sequences).

        Embeddingi słów przy użyciu GloVe (wektory 300-wymiarowe, wczytane z pliku glove.6B.300d.txt).

        TF-IDF dla modelu MLP.

    Modele:

        MLP: Sieć neuronowa z warstwami gęstymi i regularyzacją.

        CNN: Sieć konwolucyjna 1D z embeddingami GloVe.

        BiLSTM: Dwukierunkowa sieć LSTM.

    Strategia treningu:

        Balansowanie klas (class_weight).

        Wczesne zatrzymanie (EarlyStopping).

    Ewaluacja:

        Metryki: Dokładność (accuracy), raport klasyfikacji.

        Wizualizacja: Macierze pomyłek i krzywe uczenia.


## 3. Źródła Danych i Zasoby

    Boston Housing:

        Modyfikacja oryginalnego zbioru z Kaggle, dostępna na GitHub Gist. (https://gist.github.com/nnbphuong/def91b5553736764e8e08f6255390f37)

    LIAR (Fake News):

        Pobrany z repozytorium mdepak. (https://github.com/mdepak/fake-news-detection-resources)

    GloVe Embeddings:

        Oficjalne embeddingi Stanford: glove.6B.300d.txt. (https://nlp.stanford.edu/projects/glove/)

## 4. Wnioski

    House Prices Prediction:
    Modele zespołowe (Random Forest, XGBoost) osiągają lepszą wydajność niż modele liniowe dzięki uwzględnieniu nieliniowych zależności.

    Fake News Detection:
    Sieci neuronowe (szczególnie BiLSTM) skutecznie wychwytują kontekst tekstu, ale wyniki wskazują na potrzebę lepszego balansowania danych lub augmentacji.

    Ogólnie:
    Projekty pokazują, jak różne techniki przetwarzania danych (standaryzacja, embeddingi) i modele mogą być dostosowane do problemów strukturalnych (regresja) i tekstowych (klasyfikacja).
