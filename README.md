# Amharic Character Recognition
# Table of Contents
1. [Introduction](##introduction)
2. [Motivation](##motivation)
3. [Dataset](##dataset)
4. [Methodology](##methodology)
    - [Data Loading and Preprocessing](###data-loading-and-preprocessing)
    - [Train and Test Data Splitting](###Train-and-Test-Data-Splitting)
    - [Dimensionality Reduction](###dimensionality-reduction)
        - [PCA](####PCA)
        - [LDA](####LDA)
    - [Model Training Evaluation and Analysis ](###model-training)
5. [Results](##results)
6. [Usage](##usage)
7. [Contributors](##contributors)


## Introduction

This project aims to recognize Amharic characters using machine learning techniques. Its primary objective is to develop a model capable of accurately classifying Amharic characters from images. The project employs Principal Component Analysis (PCA) and Linear Discriminant Analysis (LDA) for dimensionality reduction. Additionally, it utilizes various classifiers such as Support Vector Machines (SVM), Logistic Regression, and K-nearest neighbors (KNN) for classification. The performance of each classifier is evaluated before and after applying feature extraction techniques. 

## Motivation
The motivation behind this project is to bridge the gap in existing character recognition systems that primarily focus on Latin characters. Amharic is one of the most widely spoken languages in Ethiopia, and having accurate character recognition systems can facilitate tasks such as text processing, language learning, and document digitization for Amharic speakers.

## Dataset
The dataset consists of images of handwritten Amharic characters. Each image is labeled with the corresponding character it represents. The dataset has been preprocessed to ensure consistency in image size and format, making it suitable for training machine learning models. It comprises 4200 characters, including some augmented images, with a total of 14 distinct characters for classification.

## Methodology
### Data Loading and Preprocessing
The images are loaded and preprocessed to convert them into a format suitable for training machine learning models. This includes resizing, normalization, and flattening of the image data, as well as shuffling.
```python
# Function to load images and labels
def load_images_and_labels(dataset_dir):
    data = []
    labels = []
    for root, _, files in os.walk(dataset_dir):
        label = os.path.basename(root)
        for file in files:
            with Image.open(os.path.join(root, file)) as img:
                img_resized = img.resize((64, 64)).convert('L')
                img_array = np.array(img_resized).flatten()
                data.append(img_array)
                labels.append(label)
    combined = list(zip(data, labels))
    random.shuffle(combined)
    data[:], labels[:] = zip(*combined)
    return data, labels

# Normalize data
def normalize_data(data):
    data = np.array(data)
    max_val = np.max(data)
    min_val = np.min(data)
    normalized_data = (data - min_val) / (max_val - min_val)
    return normalized_data

# Load and preprocess data
dataset_dir = "/home/hailemicael/ml_pro/dataset"
data, labels = load_images_and_labels(dataset_dir)
data = normalize_data(data)
```

### Train and Test Data Splitting
The dataset is divided into two subsets: the training set and the testing set. The training set is used to train the machine learning models, while the testing set is used to evaluate their performance on unseen data. In this project number of samples in the training set is  3360  and the number of samples in the testing set is 840.
sample images of testing set
![Test Image](https://github.com/Hailemicael/Amharic-Character-Recognition-with-ML/raw/master/test_image.png)


### Dimensionality Reduction
PCA and LDA techniques are applied to reduce the dimensionality of the image data while preserving important features. This helps in improving computational efficiency and reducing overfitting.

#### PCA (Principal Component Analysis)

PCA is a statistical method used to reduce the dimensionality of data while retaining most of its variation. In this project, PCA is employed to transform the image data into a lower-dimensional space.  Image data typically consists of high-dimensional feature vectors representing pixels or other image characteristics. By applying PCA, we can identify the principal components that capture the most significant variations in the image data. These principal components form a new set of basis vectors, allowing us to represent the images in a lower-dimensional space while preserving as much of the original information as possible.

The lower-dimensional representation obtained through PCA can be useful for various tasks such as visualization, feature extraction, and dimensionality reduction. In this project, PCA is utilized as a preprocessing step to reduce the complexity of the image data while preserving essential information, making it more manageable for the classification.
Python:

```python
def apply_pca(data, alpha=0.95):
    mean = np.mean(data, axis=0)
    centered_data = data - mean

    cov_matrix = np.dot(centered_data.T, centered_data) / (centered_data.shape[0] - 1)
    eig_values, eig_vectors = np.linalg.eigh(cov_matrix)

    idx = np.argsort(eig_values)[::-1]
    eig_values = eig_values[idx]
    eig_vectors = eig_vectors[:, idx]

    evr = eig_values / np.sum(eig_values)
    cvr = np.cumsum(evr)
    k = np.argmax(cvr >= alpha) + 1

    print(f"Using {k} components to retain {cvr[k-1]*100:.2f}% of the variance")
    reduced_data = np.dot(centered_data, eig_vectors[:, :k])
    return reduced_data, eig_vectors, k

# Apply PCA to training data
transformed_x_train, eig_vectors_pca_train, k = apply_pca(X_train, alpha=0.95)

# Apply PCA to testing data using the eigenvectors obtained from training data
centered_x_test = X_test - np.mean(X_train, axis=0)
transformed_x_test = np.dot(centered_x_test, eig_vectors_pca_train[:, :k])
```
Visualization of PCA Analyzed sample images:
![PCA Analyzed Image](https://github.com/Hailemicael/Amharic-Character-Recognition-with-ML/blob/master/Images%20.png)


#### LDA (Linear Discriminant Analysis)

LDA is a dimensionality reduction technique used in supervised learning tasks to find the linear combinations of features that best separate different classes in the data. In contrast to PCA, which focuses on maximizing the variance in the data, LDA aims to maximize the separation between classes.

In this project, LDA is applied to the transformed image data obtained after PCA. By projecting the data onto a lower-dimensional subspace defined by the most discriminative components, LDA helps enhance the class separability and improve the performance of classification algorithms.
```python
def LDA(data, labels, k=1):
    # Implementation code here...

lda_space = LDA(transformed_x_train, y_train, k=100) 
train_lda_projected = np.dot(transformed_x_train, lda_space)
test_lda_projected = np.dot(transformed_x_test, lda_space)
```
## Model Training, Evaluation, and Analysis
The training phase involves the utilization of various classifiers, including SVM, Logistic Regression, and KNN, to handle the multi-class classification tasks presented by Amharic character recognition. These classifiers are chosen for their effectiveness in handling such tasks.

After the transformation of the data through PCA and subsequent application of LDA, the trained models are evaluated using performance metrics such as accuracy, precision, and F1-score. The evaluation is performed individually for each model: SVM, KNN, and Logistic Regression.

Additionally, confusion matrices and classification reports are generated to provide deeper insights into the models' performance across different classes of Amharic characters.

## Results
The results of the experiments demonstrate the effectiveness of the proposed approach in accurately classifying Amharic characters. Each model, whether trained before PCA, after PCA, or after LDA, achieves high accuracy and demonstrate improvement in handling variations in handwriting styles and character shapes.

![confusion_matrix_SVM_after apply PCA ](https://github.com/Hailemicael/Amharic-Character-Recognition-with-ML/blob/master/confusion_matrix_SVM_after%20applying%20PCA.png)

## Usage
This project is particularly beneficial for supporting handwritten recognition systems in developing countries, where familiarity with technology is crucial. By accurately recognizing handwritten Amharic characters, it helps bridge the technological gap and promotes accessibility to advanced technologies.

Its focus on the Amharic script addresses the specific needs of regions where this language is prevalent. Leveraging machine learning techniques, it offers efficient recognition capabilities, enhancing communication, education, and accessibility for Amharic-speaking communities.

Furthermore, its open-source nature encourages collaboration and innovation, allowing developers and researchers to contribute to its improvement and expansion. This collaborative effort fosters technological advancement and literacy in Amharic-speaking regions.


## Contributors
Hailemicael Lulseged Yimer
