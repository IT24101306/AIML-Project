import joblib
import scipy.sparse as sp

# Load models
classifier = joblib.load("../models/classifier.pkl")
kmeans = joblib.load("../models/kmeans_model.pkl")
svd = joblib.load("../models/svd_reducer.pkl")
cluster_map = joblib.load("../models/cluster_label_mapping.pkl")

tfidf = joblib.load("../feature_files/tfidf_vectorizer.pkl")
tags = joblib.load("../feature_files/tags_vectorizer.pkl")
length_stats = joblib.load("../feature_files/length_stats.pkl")


# ---------------------------------------------------
# Clean prediction functions
# ---------------------------------------------------

def predict_sentiment(features):
    return classifier.predict(features)[0]


def predict_cluster(features):
    reduced = svd.transform(features)
    return kmeans.predict(reduced)[0]


def map_cluster(cluster_id):
    return cluster_map[cluster_id]