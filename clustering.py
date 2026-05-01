import os
import joblib
import numpy as np
import scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score

# ==========================================================
# FOLDERS
# ==========================================================

os.makedirs("models",exist_ok=True)
os.makedirs("models/plots",exist_ok=True)

print("="*60)
print("K-MEANS - HOTEL COMPLAINT CLUSTERING SYSTEM")
print("="*60)

# ==========================================================
# LOAD FEATURES
# ==========================================================

X = sp.load_npz("feature_files/final_feature_matrix.npz")
sentiment_labels = np.load("feature_files/sentiment_labels.npy")

label_encoder = joblib.load(
    "feature_files/label_encoder.pkl"
)

tfidf_vectorizer = joblib.load(
    "feature_files/tfidf_vectorizer.pkl"
)

feature_names = tfidf_vectorizer.get_feature_names_out()

print("\nFeature shape:",X.shape)

# ==========================================================
# FILTER NEGATIVE REVIEWS
# ==========================================================

negative_idx = list(
    label_encoder.classes_
).index("Negative")

mask = sentiment_labels==negative_idx

X_negative = X[mask]

print("Negative reviews:",X_negative.shape[0])

# ==========================================================
# SVD REDUCTION
# ==========================================================

print("\nRunning TruncatedSVD...")

svd_reducer = TruncatedSVD(
    n_components=300,
    random_state=42
)

X_reduced = svd_reducer.fit_transform(
    X_negative
)

X_reduced = normalize(X_reduced)

explained_variance = (
    svd_reducer.explained_variance_ratio_.sum()*100
)

print(
f"Explained variance: {explained_variance:.2f}%"
)

joblib.dump(
    svd_reducer,
    "models/svd_reducer.pkl"
)

# ==========================================================
# ELBOW + SILHOUETTE
# ==========================================================

print("\nTesting K values...")

sample_size=min(15000,X_reduced.shape[0])

np.random.seed(42)

sample_idx=np.random.choice(
    X_reduced.shape[0],
    sample_size,
    replace=False
)

sample=X_reduced[sample_idx]

cluster_range=range(2,9)

inertia_vals=[]
sil_vals=[]

for k in cluster_range:

    km=KMeans(
        n_clusters=k,
        n_init=10,
        random_state=42
    )

    labels=km.fit_predict(sample)

    inertia_vals.append(
        km.inertia_
    )

    sil=silhouette_score(
        sample,
        labels
    )

    sil_vals.append(sil)

# plot
plt.figure(figsize=(8,5))
plt.plot(cluster_range,inertia_vals,marker="o")
plt.title("Elbow Method")
plt.savefig(
"models/plots/elbow_curve.png",
dpi=150
)
plt.close()

# ==========================================================
# FINAL KMEANS
# ==========================================================

print("\nTraining final KMeans...")

kmeans=KMeans(
    n_clusters=4,
    n_init=20,
    max_iter=500,
    random_state=42
)

cluster_labels=kmeans.fit_predict(
    X_reduced
)

joblib.dump(
    kmeans,
    "models/kmeans_model.pkl"
)

print("K-Means trained successfully")

# ==========================================================
# EVALUATION
# ==========================================================

eval_size=min(
10000,
X_reduced.shape[0]
)

eval_idx=np.random.choice(
X_reduced.shape[0],
eval_size,
replace=False
)

sil_score=silhouette_score(
X_reduced[eval_idx],
cluster_labels[eval_idx]
)

print("\nEVALUATION")
print("-"*50)
print("Inertia:",kmeans.inertia_)
print("Silhouette:",round(sil_score,4))

# ==========================================================
# TOP WORDS PER CLUSTER
# FIXED INDEX ERROR HERE
# ==========================================================

components = svd_reducer.components_

centers_original=np.dot(
kmeans.cluster_centers_,
components
)

cluster_top_words={}

review_feature_count=len(feature_names)

for i in range(4):

    center=centers_original[i]

    # IMPORTANT FIX:
    # use ONLY review tfidf dimensions
    review_part=center[:review_feature_count]

    idx=np.argsort(
        review_part
    )[-15:][::-1]

    cluster_top_words[i]=[
        feature_names[j]
        for j in idx
    ]

# ==========================================================
# AUTO LABEL CLUSTERS
# ==========================================================

food_kw={
"food","breakfast","meal","restaurant","dining"
}

staff_kw={
"staff","service","reception","rude","manager"
}

room_kw={
"room","bathroom","bed","dirty","noise"
}

def score(words,keywords):
    return sum(
        1 for w in words
        if any(k in w for k in keywords)
    )

cluster_scores={}

for cid in range(4):

    words=cluster_top_words[cid]

    cluster_scores[cid]={
        "Food":score(words,food_kw),
        "Staff":score(words,staff_kw),
        "Rooms":score(words,room_kw)
    }

cluster_map={}
used=set()

for cid in range(4):

    scores=cluster_scores[cid]

    best=max(
        scores,
        key=scores.get
    )

    if best not in used:
        cluster_map[cid]=best
        used.add(best)

remaining=[
x for x in
["Food","Staff","Rooms","Other"]
if x not in used
]

for cid in range(4):
    if cid not in cluster_map:
        cluster_map[cid]=remaining.pop(0)

print("\nCluster Mapping")
for cid,label in cluster_map.items():
    print(
    cid,
    "->",
    label
    )

joblib.dump(
cluster_map,
"models/cluster_label_mapping.pkl"
)

# ==========================================================
# TOP WORDS PLOT
# ==========================================================

fig,axes=plt.subplots(
2,2,
figsize=(14,10)
)

axes=axes.flatten()

for cid in range(4):

    words=cluster_top_words[cid][:10]

    scores=np.arange(
        len(words),
        0,
        -1
    )

    axes[cid].barh(
        words[::-1],
        scores[::-1]
    )

    axes[cid].set_title(
    f"Cluster {cid}: {cluster_map[cid]}"
    )

plt.tight_layout()
plt.savefig(
"models/plots/cluster_top_words.png",
dpi=150
)
plt.close()

# ==========================================================
# PCA VISUAL
# ==========================================================

sample_viz=min(
5000,
X_reduced.shape[0]
)

viz_idx=np.random.choice(
X_reduced.shape[0],
sample_viz,
replace=False
)

X_viz=X_reduced[viz_idx]
y_viz=cluster_labels[viz_idx]

pca=PCA(
n_components=2,
random_state=42
)

pts=pca.fit_transform(
X_viz
)

plt.figure(figsize=(10,7))

for cid in range(4):
    m=y_viz==cid
    plt.scatter(
        pts[m,0],
        pts[m,1],
        s=8,
        alpha=.4,
        label=cluster_map[cid]
    )

plt.legend()
plt.title(
"KMeans Complaint Clusters"
)

plt.savefig(
"models/plots/cluster_visualization.png",
dpi=150
)
plt.close()

# ==========================================================
# SAVE REPORT
# ==========================================================

report_lines=[
"="*60,
"KMEANS CLUSTERING REPORT",
"="*60,
f"Negative Reviews: {X_negative.shape[0]}",
f"Features: {X.shape[1]}",
f"Inertia: {kmeans.inertia_}",
f"Silhouette: {sil_score}",
"",
"Cluster Mapping:"
]

for cid,label in cluster_map.items():

    cnt=np.sum(
        cluster_labels==cid
    )

    report_lines.append(
        f"Cluster {cid} -> {label} ({cnt})"
    )

report_lines.append("")
report_lines.append(
"Top Words:"
)

for cid in range(4):

    report_lines.append(
f"Cluster {cid}: "
+", ".join(
cluster_top_words[cid][:10]
)
)

with open(
"models/kmeans_report.txt",
"w",
encoding="utf-8"
) as f:
    f.write(
        "\n".join(
            report_lines
        )
    )

print("\nSaved Files")
print("-"*40)
print("models/kmeans_model.pkl")
print("models/svd_reducer.pkl")
print("models/cluster_label_mapping.pkl")
print("models/kmeans_report.txt")
print("models/plots/elbow_curve.png")
print("models/plots/cluster_top_words.png")
print("models/plots/cluster_visualization.png")

print("\nTraining complete.")