**Deep Learning and Neural Networks Final Project**  
*Murad Abu-Gosh, Mohammad Khayyo*

---

🌟 **Summary**:
We've embarked on a fascinating journey to solve the challenge of identifying duplicate images of Dead Sea Scroll fragments. Through two cutting-edge approaches, we aimed to detect fragments that might have been captured more than once.

---

🌍 **Introduction**:  
The Dead Sea Scrolls, discovered in the 1950s, are a treasure trove of history. However, their fragmented state presented a significant challenge. Our mission? To leverage the power of neural networks to piece this jigsaw together!

---

📚 **Previous Work**:
1. *SuperGlue: Learning Feature Matching with Graph Neural Networks* - We're standing on the shoulders of giants, utilizing their pretrained models.
2. *Theoretical Foundations of t-SNE* - This paper laid the groundwork for our visual clustering efforts.

---

📷 **Data**:  
The Museum of Israel graced us with 107 images of the Dead Sea Scrolls. Our preprocessing odyssey involved segmentation, Watershed algorithms, rotations, padding, and even creating unique match pair lists.

---

🔧 **Methods**:  
Our main arsenal? SuperGLUE by Magic Leap, a powerful Graph Neural Network. We were armed with pretrained models optimized for different scenarios, as well as our custom-trained ones. In parallel, we also tried our hand with a combination of auto-encoder and t-SNE.

---

📊 **Results**:

- **(Auto-encoder + t-SNE)**: This combo helped us find pieces with similar shape and color. While we didn't find exact matches, the results were promising.
  
- **SuperGlue with Custom Dataset**: Our custom model underwent rigorous hyperparameter tuning. But even then, without the necessary ground truths, it was a tough hill to climb.
  
- **SuperGlue Pretrained Models**: These models shone, especially when identifying larger fragments within images.

---

❌ **Error Analysis**:  
While our models were promising, they had their challenges. The auto-encoder and t-SNE combo sometimes missed the intricate texts. Our custom SuperGlue model faced issues due to limited dataset size and the absence of ground truth. However, pretrained models provided substantial insights into fragment matches.

---

🔍 **Conclusions**:  
Of all the models, the pretrained SuperGlue variants stood out. Our experiments hint at the possibility that no fragment was photographed more than once, but only the Museum of Israel can validate this claim. Future work? Consider high-res images and obtaining the elusive "ground truths" for our custom dataset.

---

💼 **Are you hiring or keen on a deeper dive into our project?**  
Our work represents just the tip of the iceberg. We're brimming with ideas to push this further and would love an opportunity to discuss them. Let's chat! 🚀