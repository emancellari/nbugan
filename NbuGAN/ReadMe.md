
# NbuGAN – High-Speed Black-Box Attack for High-Resolution Adversarial Image Generation

This project implements **NbuGAN**, a strategy for generating adversarial examples against deep learning image classification models.

The attack is demonstrated on a **pre-trained ResNet-50 model trained on the ImageNet dataset**, where a clean image is perturbed to change its predicted class toward a chosen target category.

---

## 📌 Environment Setup

Create a Python environment with the following required libraries and versions:

| Library | Version | Installation Command |
|--------|---------|----------------------|
| Python | 3.8.10 | `pip install python==3.8.10` |
| NumPy | 1.24.3 | `pip install numpy==1.24.3` |
| TensorFlow | 2.13.0 | `pip install tensorflow==2.13.0` |
| Keras | 2.13.x (recommended) | `pip install keras` |
| Pillow | Latest | `pip install pillow` |
| OpenCV | Latest | `pip install opencv-python` |

> ⚠️ It is strongly recommended to create a **virtual environment or conda environment** before installing dependencies.

---

## Sample Attack Configuration

The default experiment uses the following setup:

- **Clean image:** `baseball3.JPEG`  
- **Image resolution:** `(2336 × 3504)`  
- **Target category:** `ladle`  
- **ImageNet class index:** `421`  
- **Target label confidence:** `0.9`  
- **Attacked model:** Pre-trained **ResNet-50 (ImageNet)**  
- **Perturbation constraint:**  

```text
epsilon = 4

## Running

You can simply run NbuGAN.py in any IDE.
During the process, it will report the changes in image label and its value for each epoch. 
At the end, the generated adversarial image will be saved in .png and .npy formats. 
The "Report.txt" will provide information about the results, 
including the label and label value of the adversarial image, the number of epochs, and the time taken in seconds.




