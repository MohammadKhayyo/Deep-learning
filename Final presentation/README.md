# **Inductive t-SNE via Deep Learning to Visualize Multi-label Images (2019)**

🌎 **Institutions:**
- **Instituto Tecnológico Autónomo de México (ITAM)**, Mexico City, Mexico 
- **CVM Lab - University of Geneva**, Geneva, Switzerland

---

## 📌 **Table of Contents**
1. **Introduction**
2. **What is t-SNE?**
3. **Challenges & Solutions**
4. **Model Architecture**
5. **Partial Relevance**
6. **Dataset**
7. **Visualizations**
8. **Conclusions**

---

## **1. Introduction**

Visualizing multi-label images can be challenging. Dive deep into this project to understand how a blend of t-SNE and deep learning simplifies this task, promoting clarity and ease of comprehension.

---

## **2. What is t-SNE?**

t-SNE is a machine learning algorithm focused on dimensionality reduction. It maps high-dimensional data into a lower dimension while retaining the data's local structure, making it an ideal tool for visualization.

📝 **Key Equations**:

- 𝑝_├ 𝑗┤|𝑖
- 𝑞_├ 𝑗┤|𝑖

**Essence**: 
- High-dimensional space neighborhood → 𝑝_├ 𝑗┤|𝑖
- Low-dimensional space neighborhood → 𝑞_├ 𝑗┤|𝑖

---

## **3. Challenges & Solutions**

Traditional t-SNE methods have limitations, notably being transductive. This project aims to overcome this:

**1. Inductive Capabilities**: Allows t-SNE to generalize and apply the method to unseen data.
**2. Supervised Dimensionality Reduction**: Enhances image separability, segregating irrelevant images and emphasizing related ones.

---

## **4. Model Architecture**

To achieve inductive capabilities:

1. First, use base t-SNE for low-dimensional representation estimation.
2. Train a Deep Neural Network (DNN) that learns the dimensionality reduction process.

🖼️ **Model Structure**:

- **5-layer MLP**:
  1. Input Layer: High-dimensional representations, e.g., 4094-D.
  2-4. Hidden Layers with 128, 32, 8 units respectively. ReLU activated.
  5. Output Layer: E.g., 2 units. Sigmoid activated.
- Training specifics:
  - **SGD**: Learning rate of 0.001, momentum of 0.01, and mini-batches of 30 examples.
  - **Loss**: Mean square error (MSE).
  - **Duration**: 300 epochs.

---

## **5. Partial Relevance**

Handling multi-label-multi-instance images requires considering partial relevance. It is crucial for evaluating retrieval and classification performance. The project introduces a measure of partial relevance, aiding in enhanced evaluation.

🔍 **Partial Relevance Formula**: 𝑟(𝑙_1,𝐿_2 )

---

## **6. Dataset**

**PASCAL Visual Object Classes (VOC2012)**:
- 17,125 images, 40,124 object instances.
- Divided as: 9000 (training), 4000 (validation), 4125 (testing).

---

## **7. Visualizations**

Color coding helps in distinguishing multi-instance images. A specific strategy aids in identifying similarities and differences effectively.

🎨 **Color Coding Strategy**:
1. Different color for each class.
2. Weighted average color based on class proportion in an image.

---

## **8. Conclusions**

This methodology redefines the application of t-SNE for multi-label image visualization. It surpasses traditional methods, presenting a futuristic approach to address challenges and enhance data comprehension.

---

📩 **Interested in Discussing?** 

Feel free to reach out and explore this groundbreaking work. Remember, every piece of feedback, question, or professional connection can pave the way for innovations. Looking forward to potential discussions and interviews!

---

📢 **Please note**: For a complete understanding and a visual dive into the project, check out the entire presentation, which hosts several images and graphical data representations!

---

**Special Note**: This is a brief representation of a comprehensive research project. To grasp its entirety, one would benefit from viewing the presentation with its visual aids. If you find this work intriguing and wish to explore deeper, do reach out. Let's discuss!