# 🛍 Shopper Spectrum

### AI-Powered Customer Segmentation & Product Recommendation System

Shopper Spectrum is an end-to-end Machine Learning project that helps e-commerce businesses understand customer behavior, identify valuable customers, reduce churn risk, and generate intelligent product recommendations.

Built using **RFM Analysis, K-Means Clustering, Collaborative Filtering, Streamlit, and Plotly**.

---

## 🌐 Live Demo

🚀 **Try the Application**

https://shoopkeeper-spectrum-dhopjeigjkavtzdqbkpbpb.streamlit.app/

---

# 📌 Problem Statement

Modern e-commerce businesses generate thousands of transactions every day, but understanding customer behavior and recommending relevant products remains a challenge.

This project solves two important business problems:

### 1️⃣ Customer Segmentation
Identify different customer groups based on purchasing behavior.

### 2️⃣ Product Recommendation
Recommend similar products using customer purchasing patterns.

---

# 🎯 Project Objectives

✔ Segment customers using RFM Analysis

✔ Identify High-Value Customers

✔ Detect At-Risk Customers

✔ Improve customer retention

✔ Generate product recommendations

✔ Build an interactive business dashboard

---

# 📊 Dataset Information

**Online Retail Dataset**

| Metric | Value |
|----------|----------|
| Transactions | 541,909 |
| Customers | 4,338 |
| Products | 3,877 |
| Countries | 37 |

### Dataset Features

- InvoiceNo
- StockCode
- Description
- Quantity
- InvoiceDate
- UnitPrice
- CustomerID
- Country

---

# 🧹 Data Preprocessing

The raw dataset contained missing values, cancelled orders, and invalid transactions.

### Cleaning Steps

✅ Removed missing Customer IDs

✅ Removed cancelled invoices

✅ Removed negative quantities

✅ Removed invalid prices

✅ Converted date fields

✅ Created TotalAmount feature

```python
TotalAmount = Quantity × UnitPrice
```

### After Cleaning

| Metric | Value |
|----------|----------|
| Records | 397,884 |
| Customers | 4,338 |
| Products | 3,877 |

---

# 📈 Exploratory Data Analysis (EDA)

The project includes detailed business analytics:

- Top Countries by Sales
- Best Selling Products
- Monthly Revenue Trends
- Customer Revenue Distribution
- RFM Distribution Analysis
- Cluster Visualization
- Segment Distribution

---

# 🧠 Feature Engineering

### RFM Analysis

RFM is a popular customer analytics technique.

### Recency (R)

How recently the customer made a purchase.

### Frequency (F)

How often the customer purchases.

### Monetary (M)

How much money the customer spends.

Generated features:

```python
Recency
Frequency
Monetary
```

---

# 🤖 Machine Learning Model

## Customer Segmentation

### Algorithm Used

```python
KMeans Clustering
```

### Feature Scaling

```python
StandardScaler
```

### Optimal Cluster Selection

Used:

- Elbow Method
- Silhouette Score

### Final Choice

```python
k = 4
```

---

# 👥 Customer Segments

| Segment | Description |
|----------|------------|
| 🏆 High-Value | Loyal customers with high spending |
| ⭐ Regular | Consistent and active customers |
| 🛒 Occasional | Low engagement customers |
| ⚠ At-Risk | Customers likely to churn |

---

# 🛍 Product Recommendation Engine

### Recommendation Technique

Item-Based Collaborative Filtering

### Similarity Metric

```python
Cosine Similarity
```

### Process

1. Customer selects a product
2. Similarity scores are calculated
3. Top 5 most similar products are recommended

---

# 🖥 Streamlit Dashboard

The project is deployed as an interactive Streamlit web application.

---

## 🏠 Home Page

Features:

- Project Overview
- Dataset Statistics
- Segment Distribution
- Platform Features

---

## 🛍 Product Recommender

Features:

- Product Search
- Product Selection
- Similar Product Recommendations
- Similarity Score Visualization

---

## 👥 Customer Segmentation

Features:

- RFM Inputs
- Segment Prediction
- Business Recommendations
- Interactive Gauges

---

## 📊 Customer Insights

Features:

- Recency vs Frequency Analysis
- Frequency vs Monetary Analysis
- Segment Analytics
- RFM Statistics

---

# 🏗 Project Architecture

```text
Raw Dataset
     │
     ▼
Data Cleaning
     │
     ▼
Feature Engineering
     │
     ▼
RFM Analysis
     │
     ▼
KMeans Clustering
     │
     ▼
Customer Segmentation
     │
     ├──────────────► Dashboard
     │
     ▼
Collaborative Filtering
     │
     ▼
Product Recommendations
```

---

# 🛠 Technology Stack

### Programming

- Python

### Data Processing

- Pandas
- NumPy

### Machine Learning

- Scikit-Learn

### Visualization

- Plotly

### Deployment

- Streamlit Cloud

### Model Hosting

- Hugging Face Hub

---

# 📂 Project Structure

```text
Shopper_Spectrum/
│
├── app.py
├── requirements.txt
├── Shopper_Spectrum_Notebook.ipynb
│
├── models/
│   ├── kmeans_model.pkl
│   ├── scaler.pkl
│   ├── cluster_labels.json
│   └── rfm_data.csv
│
└── README.md
```

---

# ⚠ Large Model Storage

The recommendation model exceeds GitHub file size limits.

Therefore:

- Model hosted on Hugging Face
- Automatically downloaded during startup

Model Repository:

https://huggingface.co/ijghb/shopper-spectrum-models

---

# 🚀 Local Installation

Clone repository:

```bash
git clone YOUR_REPOSITORY_URL
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run application:

```bash
streamlit run app.py
```

---

# 📈 Future Improvements

### Planned Features

- Deep Learning Recommender System
- Customer Lifetime Value Prediction
- Sales Forecasting
- Market Basket Analysis
- Real-Time Recommendations
- Personalized Marketing Campaigns

---

# 👨‍💻 Author

## Aryan Sharma

Aspiring Data Scientist | Machine Learning Enthusiast | AI Developer

### Connect With Me

- GitHub: YOUR_GITHUB_LINK
- LinkedIn: YOUR_LINKEDIN_LINK

---

⭐ If you found this project useful, consider giving it a star.
