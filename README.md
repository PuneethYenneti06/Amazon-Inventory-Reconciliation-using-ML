# Counting Items in Amazon Warehouse Bins

**UE24CS352A – Machine Learning Mini-Project**

A re-implementation of the approach in *"Amazon Inventory Reconciliation Using AI"* (Bertorello, Sripada, Dendumrongsup, Stanford University). Given a photo of a warehouse bin, the model predicts **how many items (0–5) it contains**.

> **Credit:** The problem formulation, dataset choice, model families (Logistic Regression / SVM / ResNet transfer learning) and evaluation setup are taken from the paper above. This repository is an independent, smaller-scale recreation for coursework. It is not the authors' code. The authors' own repository is linked under [Credits](#credits-and-references).

---

## 1. Problem statement

Amazon fulfillment centers store products in bins that are moved by robots. Items are sometimes misplaced during handling, so the **recorded bin inventory no longer matches the bin's actual contents**. If the number of items in a bin can be predicted from a photo, any mismatch with the recorded count can be detected and corrected.

- **Input:** one RGB photo of a bin
- **Output:** predicted item count, as a 6-class classification problem (0, 1, 2, 3, 4, 5 items)
- **Why it is hard:** bins are cluttered, items are wrapped in tape or occluded by other items, and the products come from hundreds of thousands of different SKUs. Even a human often cannot count the items reliably.

As in the paper, we restrict the problem to bins with **at most 5 items**, which gives a random-guess baseline of 1/6 ≈ 16.7%.

## 2. Dataset

**Amazon Bin Image Dataset** (AWS Open Data, public S3 bucket `aft-vbi-pds`).

- Each bin image has a metadata JSON file. The field `EXPECTED_QUANTITY` is used as the label.
- The notebook downloads images with IDs 1–60,000 and keeps only those with 0–5 items, which gives **48,206 images**.
- Images are resized to **160 × 160** and cached to a `.npz` file so a restart does not re-download.
- Stratified split with a fixed seed: **70% train / 20% validation / 10% test** (33,743 / 9,642 / 4,821).

Class counts: 0 → 1,454 · 1 → 5,483 · 2 → 10,205 · 3 → 11,948 · 4 → 10,672 · 5 → 8,444. The majority-class baseline is 24.8%.

## 3. Implementation

Everything is in one notebook: `bin_counting_colab_v2.ipynb` (written for Google Colab with a GPU).

| Step | What it does |
|---|---|
| Data | Parallel download from S3, label extraction, resize to 160×160, cache to disk |
| EDA | Class distribution and sample images |
| Classical baselines | Images average-pooled to 32×32×3, standardised, PCA (100 components, randomized solver), then Logistic Regression and an RBF SVM (C chosen on validation: 1, 10, 100). Trained on an 8,000-image subsample because SVMs scale poorly. |
| Experiment A | ResNet18 (ImageNet-pretrained), **frozen backbone**, only the new 6-way final layer is trained |
| Experiment B | ResNet18, **full fine-tuning** |
| Experiment C (optional) | ResNet34 fine-tuning. Included in the notebook but **not completed** because of compute limits, so no results are reported for it. |
| Evaluation | Accuracy, RMSE, per-class precision/recall/F1, confusion matrix, error analysis on misclassified images |
| Demo | `demo.py` loads the saved model and predicts the count for any bin photo |

**CNN training details:** AdamW (weight decay 1e-2), cosine learning-rate decay, cross-entropy with label smoothing 0.1, mixed precision, batch size 64, 15 epochs. The weights from the **best validation epoch** are kept. Augmentation consists of random horizontal flips, random shifts (±8 px, reflect padding) and brightness jitter. At prediction time, horizontal-flip test-time augmentation averages the outputs of the image and its mirror. Images are normalised with ImageNet statistics.

### Differences from the paper

| | Paper | This project |
|---|---|---|
| Images | 150K for most experiments, 324K for the best model | 48K |
| Resolution | up to 224×224 | 160×160 |
| Best CNN | ResNet34 + Adam, 324K images | ResNet18 fine-tuned, 48K images |
| Extra feature methods | Blob and HOG features | Not implemented (PCA only) |
| Learning-rate schedule | LR finder, SGDR, differential learning rates | Cosine annealing |

## 4. Results

Test-set results (4,821 images):

| Model | Test accuracy | Test RMSE |
|---|---|---|
| Random guess | 0.167 | – |
| Logistic Regression (PCA) | 0.272 | 1.433 |
| SVM RBF (PCA) | 0.295 | 1.340 |
| A: frozen ResNet18 | 0.324 | 1.318 |
| **B: fine-tuned ResNet18** | **0.486** | **1.012** |
| *Paper: SVM (PCA)* | *0.32* | – |
| *Paper: ResNet34 Adam, 324K images* | *over 0.56 (test), 0.562 (val)* | *0.90* |

Per-class test accuracy of the best model (B):

| Items in bin | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| Accuracy | 0.966 | 0.821 | 0.603 | 0.411 | 0.338 | 0.337 |

**Takeaways**
- Fine-tuning the whole network is far better than a frozen backbone or classical models on raw pixels (0.486 vs 0.324 and 0.295).
- The CNN is about 2.9× better than random guessing, in line with the paper's conclusion that CNNs clearly beat SVMs.
- Bins with 4–5 items are the hardest, as in the paper. Heavy occlusion and tape make some images nearly impossible to count.
- Our accuracy is below the paper's 56%. This is expected given the smaller dataset and lower resolution. The paper also found that more data was the biggest single gain.

## 5. Repository contents

```
.
├── bin_counting_colab_v2.ipynb   # full pipeline: data, baselines, CNN training, evaluation
├── demo.py                       # command-line inference script
├── requirements.txt              # dependencies for the demo
├── model.pt                      # trained weights (best model by validation accuracy)
├── sample_images/                # one test photo per class for the live demo
└── README.md
```

## 6. How to run

### Reproduce training (Google Colab)
1. Open `bin_counting_colab_v2.ipynb` in Google Colab.
2. Choose **Runtime → Change runtime type → GPU** (a T4 is enough).
3. Run all cells in order. The first run downloads about 60,000 metadata files and images from the public S3 bucket and takes several minutes. Later runs load the cache.
4. The last cells save `model.pt`, `demo.py`, `requirements.txt` and `sample_images/`, and zip them for download.

Set `RUN_C = False` in the Experiment C cell to skip ResNet34 and save compute.

### Run the demo locally
```bash
pip install -r requirements.txt
python demo.py sample_images/true_3_items.jpg
# sample_images/true_3_items.jpg: predicted 3 item(s)  (confidence 41%)
```
You can pass several image paths at once. The output above is only an illustration of the format, and the actual prediction and confidence depend on the model.

## Credits and references

- **Primary reference:** P. Rodriguez Bertorello, S. Sripada, N. Dendumrongsup, *"Amazon Inventory Reconciliation Using AI"*, Stanford University. This project reproduces its approach at a smaller scale. The authors' repository: <https://github.com/OneNow/AI-Inventory-Reconciliation>
- **Dataset:** Amazon Bin Image Dataset, provided by Amazon through AWS Open Data (`s3://aft-vbi-pds`).
- **Prior work cited by the paper:** E. Park, `silverbottlep/abid_challenge` (ResNet34 on the same dataset).
- **Models:** K. He, X. Zhang, S. Ren, J. Sun, *Deep Residual Learning for Image Recognition* (ResNet), with ImageNet-pretrained weights from `torchvision`.
- **Libraries:** PyTorch, torchvision, scikit-learn, NumPy, pandas, Matplotlib, Pillow.

## Team

- *Puneeth Yenneti – PES1UG24CS348*
- *Pranav Chandrashekhar – PES1UG24CS332*

Course: UE24CS352A Machine Learning, PES University.
