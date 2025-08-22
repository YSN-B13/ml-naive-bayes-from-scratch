# 🤖 Naive Bayes Classifiers from Scratch  

This repository contains two projects that implement **Naive Bayes classifiers from scratch**:  

1. 🎾 **Categorical Naive Bayes (with Laplace smoothing)** using the **Play Tennis dataset**  
2. 🌸 **Gaussian Naive Bayes** using the **Iris dataset**  

Both are implemented **without scikit-learn’s Naive Bayes functions**, relying only on **Python + NumPy**.  

---

## 📂 Project Structure
**tennis_naive_bayes :**
- 📄 play_tennis.csv      # Training dataset
- 🐍 naive_bayes.py       # Core classifier implementation
- 🧪 test_cases.py        # Prediction examples & validation

**iris_gaussian_nb :**
- 📄 iris.data.csv # Iris dataset
- 🐍 gaussian_nb.py # Gaussian Naive Bayes implementation
- 🧪 test_model.py # Train/test split + evaluation


---
# 🎾 Naive Bayes Classifier: Play Tennis Dataset  
This project implements a **Naive Bayes classifier from scratch (with Laplace smoothing)** using the classic **Play Tennis dataset**.  

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

---

# 🌸 Gaussian Naive Bayes Classifier: Iris Dataset  

This project implements a **Gaussian Naive Bayes classifier from scratch** using the **Iris dataset**.  

### 📊 Dataset  

| Sepal Length | Sepal Width | Petal Length | Petal Width | Species    |
|--------------|-------------|--------------|-------------|------------|
| 5.1          | 3.5         | 1.4          | 0.2         | Setosa     |
| 7.0          | 3.2         | 4.7          | 1.4         | Versicolor |
| 6.3          | 3.3         | 6.0          | 2.5         | Virginica  |
| ...          | ...         | ...          | ...         | ...        |  

- **Features:** Continuous (sepal_length, sepal_width, petal_length, petal_width)  
- **Target:** Species (Setosa, Versicolor, Virginica)  

---

### 🧮 Mathematical Foundation  

**Bayes Theorem:**  

$$
P(C_k|X) \propto P(C_k) \times \prod_{i} P(x_i|C_k)
$$  

**Gaussian Likelihood:**  

$$
P(x|C_k) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)
$$  

Where:  
- `μ` = mean of feature in class  
- `σ²` = variance of feature in class  

---
