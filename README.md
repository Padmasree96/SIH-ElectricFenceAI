# SIH-ElectricFenceAI
AI/ML model to detect illegal electric fence currents vs normal household currents using RMS, Pulse Rate, and Peak Current. Includes dataset, training code, and SIH 2025 PPT. Enhances electrical safety, prevents accidents, and detects power theft.

Here’s a **fully detailed README template** for your SIH Electric Fence AI/ML project --

# ⚡ Smart Illegal Electric Fence Detection – SIH 2025

Electric fences are widely used for security, but unauthorized connections or misuse can cause **electrocution risks** and **power theft**. To tackle this, I developed an **AI/ML model** that predicts **illegal current supplies** and distinguishes **normal household current vs dangerous fence current**. The model analyzes features such as **RMS (Root Mean Square), Pulse Rate, Peak Current, Frequency**, and more, effectively differentiating between household appliances like **fans, bulbs, motors** and illegal fence connections.

---

## 🔹 Key Features

* **Illegal Current Detection:** Detects attempts to draw current illegally from fences.
* **Appliance Differentiation:** Separates normal appliance currents from dangerous fence currents.
* **Machine Learning Algorithms:** Uses **Random Forest** with **train-test split** for accurate predictions.
* **IoT Integration Potential:** Can be extended for real-time monitoring using IoT sensors.
* **Safety & Theft Prevention:** Enhances electrical safety and detects power theft efficiently.

---

## 🔹 Repository Contents

* **Dataset:** Curated dataset used for training the ML model.
* **Notebook & Training Code:** For reproducing, testing, and extending the model.
* **PPT Presentation:** Presented at **Smart India Hackathon (SIH) 2025**, covering problem statement, solution, methodology, feasibility, and impact.

---

## 🔹 Installation

1. **Clone the Repository**

```bash
git clone https://github.com/Padmasree96/SIH-ElectricFenceAI.git
cd SIH-ElectricFenceAI
```

2. **Set up a Python environment** (recommended: conda or virtualenv)

```bash
conda create -n electricfence python=3.10
conda activate electricfence
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

*(Dependencies may include: pandas, numpy, scikit-learn, matplotlib, seaborn)*

---

## 🔹 Usage

1. **Load Dataset:**

```python
import pandas as pd
df = pd.read_csv("electric_fence_dataset.csv")
```

2. **Run Training Notebook:**

* Open `SIH Electric Fence.ipynb` in Jupyter or VSCode.
* Run the cells sequentially to train the model and test predictions.

3. **Predict Illegal Currents:**

```python
model.predict(new_data)  # Replace new_data with features RMS, Pulse Rate, Peak Current, Frequency
```

4. **Evaluate Accuracy:**

* Model metrics like accuracy, precision, recall, and confusion matrix are displayed in the notebook.

---

## 🔹 Sample Output

| Appliance/Fence | Predicted Label | Current Type  |
| --------------- | --------------- | ------------- |
| Fan             | Normal          | Household     |
| Electric Fence  | Illegal         | Fence Current |
| Motor           | Normal          | Household     |

*(Graphs and charts showing RMS, Pulse Rate vs Current are included in the notebook.)*

---

## 🔹 Future Work

* **Real-Time IoT Integration:** Deploy sensors to detect illegal currents in real time.
* **Advanced ML Models:** Explore deep learning for improved accuracy.
* **Mobile App Integration:** Notify authorities or homeowners immediately when illegal current is detected.
* **Data Expansion:** Collect more data for diverse appliances and fence types for better model generalization.

---

## 🔹 Acknowledgements

* **Smart India Hackathon (SIH) 2025:** Platform for ideation and presentation.
* **Dataset & Tools:** Python, scikit-learn, Jupyter Notebook.
* Grateful for the opportunity to apply ML to **real-world electrical safety** problems.

---

## 🔹 License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.
