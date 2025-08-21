# 🎾 Naive Bayes Classifier: Play Tennis Dataset

This project implements a **Naive Bayes classifier from scratch (with Laplace smoothing)** using the classic **Play Tennis dataset**.

---

## 📂 Project Structure
- 📄 play_tennis.csv      # Training dataset
- 🐍 naive_bayes.py       # Core classifier implementation
- 🧪 test_cases.py        # Prediction examples & validation


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

## 🧮 Mathematical Foundation

### Naive Bayes Formula

The classifier uses Bayes' theorem with the "naive" independence assumption:

$$
P(Play|X) \propto P(Play) \times \prod_{i} P(X_i|Play)
$$

Where:
-   `X` = {Outlook, Temperature, Humidity, Wind}
-   `P(Play)` is the class prior probability.
-   `P(Xᵢ|Play)` is the feature likelihood given the class.
  
### Laplace Smoothing

To handle zero probabilities, we apply additive (Laplace) smoothing:

$$
P(feature|class) = \frac{\text{count}(feature, class) + \alpha}{\text{count}(class) + \alpha \times K}
$$

Where:
-   `α = 1` (smoothing parameter)
-   `K` = number of unique values for the feature
