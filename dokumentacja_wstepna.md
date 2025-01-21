# Dokumentacja wstępna projektu

## Heurystyczne Algorytmy Eliminacji Nieistotnych Atrybutów na Podstawie Zależności Wyższych Rzędów

### Cel Projektu
Celem projektu jest zaimplementowanie dwóch heurystycznych algorytmów eliminacji nieistotnych atrybutów: **Stability Selection** i **Recursive Feature Elimination (RFE)**, z uwzględnieniem zależności wyższych rzędów. Algorytmy te zostaną przetestowane na zbiorze danych Breast Cancer z `scikit-learn` oraz modelu uczenia maszynowego Las Losowy.

---

### Zbiór Danych
Do testowania algorytmów wykorzystany zostanie zbiór danych **Breast Cancer** z `scikit-learn`, który zawiera informacje o cechach nowotworów piersi i ich klasyfikację jako złośliwych (1) lub łagodnych (0).

#### Charakterystyka zbioru danych:
- **30 cech** opisujących różne właściwości nowotworów, takie jak promień, tekstura, obwód, powierzchnia, itd.
- **569 próbek** (wierszy), z których każda jest przypisana do jednej z dwóch klas: 
  - 0 (łagodny)
  - 1 (złośliwy).

---

### Algorytmy

#### 1. Stability Selection
Stability Selection to algorytm, który dokonuje eliminacji cech na podstawie ich stabilności. Działanie algorytmu:
- Wielokrotne losowe próbkowanie podzbiorów danych.
- Trenowanie modelu (w tym przypadku lasów losowych) na tych próbkach.
- Cechy, które są często wybierane jako istotne w różnych próbkach, są uznawane za istotne.
- Cechy rzadko wybierane są eliminowane.

#### 2. Recursive Feature Elimination (RFE)
RFE to algorytm, który iteracyjnie eliminuje najmniej istotne cechy, oceniając ich wpływ na wynik klasyfikacji. Działanie algorytmu:
- Trenowanie modelu na pełnym zbiorze cech.
- Iteracyjne usuwanie cech o najmniejszym wpływie na wynik modelu.
- Powtarzanie procesu aż do osiągnięcia pożądanej liczby cech.

---

### Schemat Rozwiązania

1. **Przygotowanie Danych**:
   - Podzielenie zbioru danych na dane treningowe (80%) i testowe (20%).

2. **Model**:
   - **Lasy Losowe (Random Forest)** będą używane jako model bazowy dla obu algorytmów.

3. **Wykonanie Algorytmów**:
   - **Stability Selection**: Ocena stabilności cech na podstawie wielu próbek losowych, wybierając te najczęściej uznawane za istotne.
   - **RFE**: Iteracyjne eliminowanie najmniej istotnych cech aż do osiągnięcia optymalnej liczby cech.

4. **Ocena Modelu**:
   - Ocena dokładności, precyzji, recall oraz F1-score na zbiorze testowym.
   - Porównanie wyników modelu przed i po eliminacji cech.

5. **Wizualizacja**:
   - Wyniki eliminacji cech zostaną przedstawione za pomocą wykresów `matplotlib`, pokazujących ważność cech przed i po eliminacji.

---

### Eksperymenty

#### 1. Wpływ liczby cech na jakość modelu
- Wykonanie Stability Selection oraz RFE z różnymi liczbami cech.
- Ocena dokładności, precyzji, recall i F1-score dla każdej konfiguracji.
- Analiza wpływu eliminacji cech na czas trenowania i wydajność modelu.

#### 2. Stability Selection: wpływ liczby iteracji i wielkości próbek
- Testowanie algorytmu Stability Selection z różną liczbą iteracji.
- Zmiana wielkości losowych próbek danych i ocena stabilności wyboru cech.

#### 3. Porównanie algorytmów Stability Selection i RFE
- Uruchomienie obu algorytmów na tych samych danych.
- Porównanie wybranych cech, dokładności modelu oraz czasu wykonania algorytmów.
- Analiza różnic w wynikach między algorytmami.

#### 4. Eksperymenty z innymi modelami
- Przetestowanie algorytmów z innymi gotowymi modelami w celu sprawdzenia uniwersalności wybranych cech.
- Porównanie wyników z gotowymi implementacjami Stability Selection i RFE.

---

### Wykorzystywane Biblioteki
- **scikit-learn**: Używane do załadowania zbioru danych oraz trenowania modeli (lasy losowe).
- **matplotlib**: Używane do wizualizacji wyników, w tym wykresów ważności cech oraz wykresów porównawczych przed i po eliminacji cech.
- **numpy**: Używane do manipulacji danymi numerycznymi.
- **pandas**: Może być użyte do analizy danych, ale nie jest konieczne, jeżeli dane są w formacie, który `scikit-learn` potrafi bezpośrednio załadować.

---

### Wyniki
Po przeprowadzeniu eksperymentów algorytmy Stability Selection oraz RFE powinny:
- Wskazać, które cechy mają największy wpływ na klasyfikację nowotworów.
- Poprawić wydajność modelu poprzez eliminację nieistotnych cech.
- Przedstawić wyniki wizualnie w sposób zrozumiały i łatwy do analizy.