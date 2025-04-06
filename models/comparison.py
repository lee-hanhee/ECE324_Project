import matplotlib.pyplot as plt
import numpy as np
class_names = ["drums", "guitar", "piano", "bass", "weighted average"]

instrunet_f1 = [0.8376, 0.7098, 0.6630, 0.8727, 0.8106]
yamnet_f1 = [0.667, 0.706, 0.857, 1.000, 0.353]

# plot bar graph two seperate bars for each model type
plt.figure(figsize=(10, 5))
plt.bar(np.arange(len(class_names)), instrunet_f1, width=0.4, label='InstruNET', color='blue', alpha=0.7)
plt.bar(np.arange(len(class_names)) + 0.4, yamnet_f1, width=0.4, label='YAMNet', color='orange', alpha=0.7)
plt.xticks(np.arange(len(class_names)) + 0.2, class_names)
plt.ylabel('F1 Score')
plt.title('F1 Score Comparison between InstruNET and YAMNet')
plt.legend()
plt.tight_layout()
plt.show()

