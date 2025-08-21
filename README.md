# 🎾 Naive Bayes Classifier: Play Tennis Dataset

This project implements a **Naive Bayes classifier from scratch (with Laplace smoothing)** using the classic **Play Tennis dataset**.

---

## 📂 Project Structure
├── play_tennis.csv # dataset
├── naive_bayes.py # Naive Bayes implementation
├── test_cases.py # test examples
├── README.md # documentation

---

## 📊 Dataset
The dataset describes weather conditions and whether tennis was played:

| Outlook   | Temp  | Humidity | Wind   | Play |
|-----------|-------|----------|--------|------|
| Sunny     | Hot   | High     | Weak   | No   |
| Rain      | Mild  | Normal   | Weak   | Yes  |
| Overcast  | Cool  | Normal   | Strong | Yes  |
| ...       | ...   | ...      | ...    | ...  |

---

## 🧮 Naive Bayes Formula

We use:

\[
P(Play \mid X) \propto P(Play) \times \prod_i P(X_i \mid Play)
\]

Where \(X\) = {Outlook, Temp, Humidity, Wind}.  
Laplace smoothing is applied:

\[
P(feature \mid class) = \frac{count(feature, class) + 1}{count(class) + K}
\]

- \(K\) = number of categories in the feature.

---

## ▶️ Run the Code

1. Clone the repo:
   ```bash
   git clone https://github.com/yourusername/naive-bayes-play-tennis.git
   cd naive-bayes-play-tennis
2. Run test cases:
   python test_cases.py
