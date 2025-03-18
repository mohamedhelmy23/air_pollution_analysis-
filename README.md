📊 Comprehensive Air Quality Analysis (EDA) 🚀 Advanced Air Quality Analysis in India Using EDA & Machine Learning In this project, we performed an in-depth Exploratory Data Analysis (EDA) on an air quality dataset to understand the key factors influencing AQI (Air Quality Index). Our approach involved data cleaning, outlier detection, advanced visualizations, and statistical analysis using Python, Pandas, Seaborn, Matplotlib, and Scikit-learn.

🔹 1. Data Cleaning & Preprocessing ✅ 📥 Importing Libraries & Dataset python Copy Edit import pandas as pd import numpy as np import matplotlib.pyplot as plt import seaborn as sns from sklearn.impute import SimpleImputer from sklearn.preprocessing import LabelEncoder ✔️ Loaded the dataset and analyzed missing values. ✔️ Converted categorical columns (e.g., city names) into numeric format using Label Encoding. ✔️ Handled missing numerical values using Mean Imputation. ✔️ Identified and treated outliers using Tukey’s Method and IQR-based filtering.

🔹 2. Exploratory Data Analysis (EDA) 📌 Univariate Analysis ✔️ We used Density Plots, Box Plots, Violin Plots, and Q-Q Plots to analyze the distribution of variables. ✔️ High pollution spikes were observed in PM2.5, PM10, NO2, and AQI, indicating periods of severe pollution.

📊 Example: PM2.5 Distribution

python Copy Edit sns.kdeplot(df['PM2.5'], fill=True, color='royalblue') plt.title('Density Plot of PM2.5') plt.show() 📌 Bivariate Analysis ✔️ Pair Plots, Regression Plots, and Heatmaps were used to analyze relationships between variables. ✔️ PM2.5 and PM10 showed a strong positive correlation, meaning they likely originate from the same pollution sources. ✔️ AQI is strongly influenced by NO2 and NOx levels, suggesting nitrogen oxides significantly impact air quality.

📊 Example: Correlation Heatmap

python Copy Edit plt.figure(figsize=(12, 6)) sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=0.5) plt.title('Correlation Matrix') plt.show() 📌 Multivariate Analysis ✔️ 3D Scatter Plots and Clustering (K-Means) were applied to uncover hidden patterns. ✔️ The clustering results showed that highly polluted cities tend to exhibit similar levels of PM2.5, NO2, and CO.

📊 Example: K-Means Clustering

python Copy Edit from sklearn.cluster import KMeans

selected_features = df[['PM2.5', 'NO2']].dropna() kmeans = KMeans(n_clusters=3) clusters = kmeans.fit_predict(selected_features)

plt.figure(figsize=(8, 6)) plt.scatter(selected_features.iloc[:, 0], selected_features.iloc[:, 1], c=clusters, cmap='coolwarm', alpha=0.5) plt.xlabel('PM2.5') plt.ylabel('NO2') plt.title('Clustering of PM2.5 & NO2') plt.show() 🔹 3. Key Challenges & Findings 📌 Challenges Faced: ✔️ 16% missing values in AQI_Bucket, handled using Imputation. ✔️ Extreme outliers in pollution levels, requiring careful data preprocessing. ✔️ Strong correlations between variables, allowing for dimensionality reduction.

📌 Major Findings: ✔️ AQI is heavily influenced by PM2.5, NO2, and NOx levels. ✔️ High-pollution cities share similar contamination patterns, making it possible to predict AQI effectively. ✔️ Using clustering, we classified regions into three pollution groups, which can guide environmental policies.

🔹 4. Conclusion & Next Steps ✅ We successfully explored and visualized key air pollution factors using advanced EDA techniques. ✅ These insights can be used to build AI-powered predictive models for air pollution forecasting. ✅ Next, we will develop a Machine Learning model to predict AQI levels using deep learning techniques.
