# Movie Recommendation System - Graph Neural Network Based Approach

## 📌 Introduction
This project implements a **graph-based movie recommendation system** using **Graph Neural Networks (GNNs)** and **A* search**. It leverages **semantic embeddings** from transformers and structured movie metadata to generate personalized recommendations.

## 📜 Overview
The system processes the **TMDB Movie Dataset** (70K/930K movies) to build a graph, where nodes represent movies, and edges signify relationships based on textual similarity and metadata attributes.

The project integrates multiple methodologies:
- **Graph Construction:** Movies are modeled as nodes in a graph, where edges represent similarity based on a combination of scalar and textual features.
- **Graph Neural Networks (GNN):** A GraphSAGE-based network is used to learn robust embeddings from the constructed graph.
- **Clustering:** KMeans clustering is applied to combined features to group similar movies and improve recommendation quality.
- **Heuristic Search (A*):** An A* search algorithm leverages computed heuristics on the graph for exploring potential recommendations.
- **Reinforcement Learning (RL):** An RL agent refines recommendations by incorporating user feedback, ensuring that the system adapts to the user's preferences over time.

## ⚙️ How It Works

### 1️⃣ Data Processing
- The dataset is preprocessed, including **feature extraction**, **embedding generation**, and **graph construction**.
- Movies are encoded with scalar attributes (e.g., language, year) and **textual embeddings** (overview, tagline, genres).

### 2️⃣ Graph Construction
- A **graph is built using NetworkX**, where:
- Nodes represent movies.
- Edges are added based on **cosine similarity** of embeddings and metadata.
- Each movie node is enriched with **feature vectors** derived from a transformer model.

### 3️⃣ Graph Neural Network (GNN) Training
- A **GraphSAGE-based model** is trained to learn representations of movies.
- The model minimizes classification loss to improve embeddings.

### 4️⃣ Movie Recommendation
- **GNN-based recommendations**: Finds movies with the most similar learned embeddings.
- **A* search-based recommendations**: Uses a priority queue to explore similar movies heuristically.

### 5️⃣ Reinforcement Learning:
   - Use an RL agent to select the final set of recommendations by considering the user context and feedback.
   - Collect user feedback and update the model, thereby continuously refining future recommendations.

## 🏗️ Setup & Installation
### 1️⃣ Prerequisites
Ensure you have Python 3.11+ installed and required dependencies:
```sh
pip install networkx pandas torch transformers torch_geometric scikit-learn numpy nltk
```

### 2️⃣ Dataset
Download the TMDB dataset from [Kaggle](https://www.kaggle.com/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies) and place it in `datasets/final-ds.csv`.

### 3️⃣ Precomputed Embeddings
Load the **precomputed embeddings** (if available):
```python
with open("embeds.pkl", "rb") as embfile:
embs = pkl_load(embfile)
```
Otherwise, generate embeddings using **Sentence Transformers**.

## 🚀 Running the System
1. **Train the Model**
 ```python
 acc_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 recommender = GraphRecommender(mov_dataset, device=acc_device)
 recommender.train()
 ```

2. **Get Recommendations**
 ```python
 inp_titles = []
 inp = input("Enter a title (leave blank to stop) >>> ")
 inp_titles.append(inp)
 while inp.strip() != "":
 inp = input("Enter a title >>> ")
 inp_titles.append(inp)
 recommender.get_recommendations(inp_titles)
 ```

## 🧠 Key Features
✅ **Graph-based Movie Similarity** using embeddings and metadata
✅ **Graph Neural Network (GNN)** for learning representations
✅ **A* Search Algorithm** for efficient recommendation
✅ **Transformer Embeddings** for textual features
✅ **Clustering for Efficiency** using K-Means

## 📝 Future Improvements
- **Hyperparameter Optimization** for GNN training
- **Leveraging Relational Graph Attention Networks (RGAT):** Investigate replacing GraphSAGE with RGAT to dynamically learn attention weights for different relationships in the movie graph to better capture the nuances of inter-movie connections, leading to improved embedding quality and more personalized recommendations.
- **Integration with External APIs** like IMDb for better results
- **Interactive Web Interface** for user-friendly experience
