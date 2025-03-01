# **Deep Detect: Advanced Anomaly Detection System**

## **Overview**
This project is an advanced anomaly detection system designed to identify subtle anomalies in high-dimensional, noisy datasets. It leverages a deep autoencoder model to detect deviations from normal behavior and provides contextual insights into the detected anomalies. The system is built using TensorFlow for model training and Streamlit for deployment and visualization.

---

## **Features**
1. **Subtle Anomaly Detection**:
   - Detects anomalies that deviate slightly from normal behavior, even in noisy data.
   - Uses a deep autoencoder model for robust pattern recognition.

2. **Contextual Insights**:
   - Provides domain-specific interpretation of anomalies.
   - Differentiates between benign variations and potential risks.

3. **Scalable and Efficient**:
   - Processes large datasets in chunks to handle memory constraints.
   - Dynamically adjusts anomaly thresholds based on data patterns.

4. **Interactive Visualization**:
   - Visualizes reconstruction errors and anomaly distributions.
   - Provides feature importance analysis to understand contributing factors.

---

## **Dataset**
The project uses the **Tennessee Eastman Process (TEP) Dataset**, which is a simulated industrial process control dataset with faults for anomaly detection. The dataset can be downloaded from [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/6C3JR1).

The dataset includes:
- `TEP_FaultFree_Training.csv`: Training data with normal behavior.
- `TEP_FaultFree_Testing.csv`: Testing data with normal behavior.
- `TEP_Faulty_Training.csv`: Faulty data for anomaly detection.
- `TEP_Faulty_Testing.csv`: Faulty data for anomaly detection.

---

## **Installation**
1. Clone the repository:
   ```bash
   git clone https://github.com/djdiptayan1/AnomalyDetection.git
   cd AnomalyDetection
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the dataset from the [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/6C3JR1) and place the CSV files in the `data/` folder.

---

## **Usage**

### 1. Train the Model
To train the autoencoder model, you can either:

- Run the Jupyter notebook:
  ```bash
  jupyter notebook train.ipynb
  ```
- Or run the Python script:
  ```bash
  python train_model.py
  ```

This will:
- Preprocess the data.
- Train the autoencoder model.
- Save the trained model (`autoencoder.h5`) and scaler (`scaler.npy`) in the `models/` folder.

### 2. Detect Anomalies
To detect anomalies in a dataset, you can either:

- Run the Jupyter notebook:
  ```bash
  jupyter notebook detect.ipynb
  ```
- Or run the Python script:
  ```bash
  python detect_anomalies.py
  ```

This will:
- Load the trained model and scaler.
- Process the dataset in chunks to detect anomalies.
- Save the results with anomaly flags in `data/TEP_Faulty_Testing_Anomalies.csv`.

### 3. Run the Streamlit App
To launch the interactive Streamlit app, run:

```bash
streamlit run app.py
```

This will:
- Start a local server and open the app in your browser.
- Allow you to upload datasets, detect anomalies, and visualize results interactively.

---

## **Methodology**

### **Data Preprocessing**:
- The dataset is normalized using `MinMaxScaler` to ensure all features are on the same scale.
- The first column (index) is dropped if present.

### **Model Architecture**:
A deep autoencoder is used to learn the normal behavior of the data. The model consists of:
- **Input layer**: Same size as the number of features.
- **Hidden layers**: Two dense layers with ReLU activation.
- **Output layer**: Reconstructs the input with sigmoid activation.

### **Anomaly Detection**:
- The reconstruction error (Mean Squared Error) is calculated for each data point.
- Anomalies are flagged if the reconstruction error exceeds a dynamic threshold (mean + 3 × standard deviation).

### **Visualization**:
- Reconstruction error distribution is plotted as a histogram.
- Anomalies are visualized over time (if a time column exists).
- Feature importance is analyzed to understand contributing factors.

---

## **Deployment**
The project is deployed using Streamlit, which provides an interactive web interface for anomaly detection. To deploy the app:

1. Install Streamlit:
   ```bash
   pip install streamlit
   ```
2. Run the app:
   ```bash
   streamlit run app.py
   ```
3. Access the app via the provided URL in your browser.

---

## **Results**
The system provides:
- **Accurate anomaly detection**: Identifies subtle deviations in noisy data.
- **Contextual insights**: Differentiates between benign variations and critical anomalies.
- **Scalable processing**: Handles large datasets efficiently using chunk-based processing.

---

## **Team Members**
- Sushant Chavan
- Diptayan Jash
- Gaurav Mishra
- Utkarsh Jaiswal

---

## **Acknowledgments**
- **Tennessee Eastman Process Dataset**: Provided by Harvard Dataverse.
- **TensorFlow**: For building and training the autoencoder model.
- **Streamlit**: For deploying the interactive web app.

---

## **Contact**
For questions or feedback, please contact sushantchavan920@gmail.com.
