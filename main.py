import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

# ML tools
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# Set page configuration
st.set_page_config(
    page_title="Iris Species Classification & Analytics System",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling (Glassmorphism, dark slates, HSL highlights, Outfit typography)
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap" rel="stylesheet">

<style>
    /* Main app body background and font */
    .stApp {
        background-color: #0b0f19;
        color: #e2e8f0;
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Headers style */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 600 !important;
        color: #ffffff;
    }

    /* Professional gradient title */
    .hero-title {
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
        font-size: 2.8rem !important;
        margin-bottom: 0.5rem;
        text-align: center;
        letter-spacing: -0.025em;
    }

    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.15rem;
        text-align: center;
        margin-bottom: 2.5rem;
        font-weight: 400;
    }

    /* Glassmorphism card layouts */
    .glass-card {
        background: rgba(30, 41, 59, 0.45);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.75rem;
        margin-bottom: 1.5rem;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        transition: transform 0.3s ease, border-color 0.3s ease;
    }

    .glass-card:hover {
        transform: translateY(-2px);
        border-color: rgba(99, 102, 241, 0.4);
    }

    /* Section subtitle styling */
    .section-title {
        color: #818cf8;
        font-weight: 600;
        font-size: 1.25rem;
        margin-top: 0.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #6366f1;
        padding-left: 0.75rem;
    }

    /* Output result badges */
    .badge {
        display: inline-block;
        padding: 0.35rem 1rem;
        border-radius: 9999px;
        font-size: 0.9rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.075em;
    }

    .badge-setosa {
        background-color: rgba(99, 102, 241, 0.15);
        color: #818cf8;
        border: 1px solid rgba(99, 102, 241, 0.35);
    }

    .badge-versicolor {
        background-color: rgba(236, 72, 153, 0.15);
        color: #f472b6;
        border: 1px solid rgba(236, 72, 153, 0.35);
    }

    .badge-virginica {
        background-color: rgba(20, 184, 166, 0.15);
        color: #2dd4bf;
        border: 1px solid rgba(20, 184, 166, 0.35);
    }

    /* Custom style for tab navigation */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background-color: rgba(15, 23, 42, 0.65);
        padding: 8px;
        border-radius: 14px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }

    .stTabs [data-baseweb="tab"] {
        height: 48px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 8px;
        color: #94a3b8;
        font-weight: 500;
        border: none;
        padding: 0 20px;
        transition: all 0.25s ease;
    }

    .stTabs [data-baseweb="tab"]:hover {
        color: #ffffff;
        background-color: rgba(255, 255, 255, 0.05);
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%) !important;
        color: #ffffff !important;
        font-weight: 600;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# Main Title & Header Section
st.markdown("<h1 class='hero-title'>Iris Species Classification & Analytics System</h1>", unsafe_allow_html=True)
st.markdown("<p class='hero-subtitle'>A Professional Multi-Model Benchmarking & Real-time Inference System</p>", unsafe_allow_html=True)

# Load raw Iris dataset
@st.cache_data
def get_dataset():
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    df['species_name'] = df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
    return iris, df

iris, df = get_dataset()

# Setup system tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Exploratory Data Analytics (EDA)",
    "⚙️ Model Tuning & Benchmarking",
    "🔮 Real-time Inference Engine",
    "📋 System Architecture & Math"
])

# ==============================================================================
# TAB 1: EXPLORATORY DATA ANALYTICS (EDA)
# ==============================================================================
with tab1:
    st.markdown("<div class='section-title'>Exploratory Data Analysis & Descriptive Statistics</div>", unsafe_allow_html=True)

    col_stats, col_charts = st.columns([1, 1.2])

    with col_stats:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("#### 📈 Statistical Summary")
        st.write("Descriptive features statistical analysis (N=150 samples):")
        st.dataframe(df.drop(['target', 'species_name'], axis=1).describe().round(2), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("#### 🌸 Dataset Overview")
        st.write(
            "The Iris dataset is a classical botanical database introduced by the British statistician Ronald Fisher in 1936. "
            "It consists of **150 records** split equally across 3 distinct species. The task is a multi-class classification "
            "problem mapping 4 input dimensions to the categorical species class label."
        )
        st.write("**Feature Vector Dimensions:**")
        st.markdown(
            "- **Sepal Length (cm)**: Range 4.3 to 7.9 cm\n"
            "- **Sepal Width (cm)**: Range 2.0 to 4.4 cm\n"
            "- **Petal Length (cm)**: Range 1.0 to 6.9 cm\n"
            "- **Petal Width (cm)**: Range 0.1 to 2.5 cm"
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col_charts:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("#### 📊 Diagnostic Visualization Panel")

        eda_choice = st.selectbox(
            "Select Diagnostic Plot",
            ["Pearson Correlation Heatmap", "Morphological Distributions (Boxplot)", "Petal Dimensions (Scatter)", "Global Pairwise Matrix (Pairplot)"]
        )

        if eda_choice == "Pearson Correlation Heatmap":
            fig, ax = plt.subplots(figsize=(6, 4))
            fig.patch.set_facecolor('#0f172a')
            ax.set_facecolor('#1e293b')
            sns.heatmap(df.drop(['target', 'species_name'], axis=1).corr(),
                        annot=True, cmap='coolwarm', fmt=".2f", ax=ax, cbar_kws={'label': 'Correlation coeff'})
            ax.tick_params(colors='white')
            for text in ax.texts:
                text.set_color('white')
            plt.title("Pearson Correlation Matrix", color='white', pad=10)
            st.pyplot(fig)
            st.caption("Insight: Petal length and petal width show an extremely strong positive correlation of 0.96.")

        elif eda_choice == "Morphological Distributions (Boxplot)":
            fig, ax = plt.subplots(figsize=(6, 4))
            fig.patch.set_facecolor('#0f172a')
            ax.set_facecolor('#1e293b')
            sns.boxplot(data=df, x='species_name', y='petal length (cm)', palette='muted', ax=ax)
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            plt.title("Petal Length Variance by Species", color='white', pad=10)
            st.pyplot(fig)
            st.caption("Insight: Petal length provides clear class separation, with Setosa having short petals, and Virginica having the longest.")

        elif eda_choice == "Petal Dimensions (Scatter)":
            fig, ax = plt.subplots(figsize=(6, 4))
            fig.patch.set_facecolor('#0f172a')
            ax.set_facecolor('#1e293b')
            sns.scatterplot(data=df, x='petal length (cm)', y='petal width (cm)',
                            hue='species_name', palette='Set1', s=80, alpha=0.9, ax=ax)
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            leg = ax.legend()
            for text in leg.get_texts():
                text.set_color('white')
            plt.title("Petal Length vs Petal Width Scatterplot", color='white', pad=10)
            st.pyplot(fig)
            st.caption("Insight: Setosa is perfectly linearly separable from the other two classes using petal measurements.")

        elif eda_choice == "Global Pairwise Matrix (Pairplot)":
            pairplot_path = "outputs/iris_pairplot.png"
            if os.path.exists(pairplot_path):
                st.image(pairplot_path, caption="Pre-computed Pairwise Distributions (Pairplot Grid)")
            else:
                st.info("Global Pairplot image is generated during training runs. Please run training inside the CLI script or Benchmarking tab to view.")

        st.markdown("</div>", unsafe_allow_html=True)

# ==============================================================================
# TAB 2: MODEL TUNING & BENCHMARKING
# ==============================================================================
with tab2:
    st.markdown("<div class='section-title'>Interactive Model Optimization and Evaluation</div>", unsafe_allow_html=True)

    # Stratified Train-Test split setup (80% Train, 20% Test)
    X = df.drop(['target', 'species_name'], axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    col_tuning, col_results = st.columns([1, 1.2])

    with col_tuning:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("#### ⚙️ Hyperparameter Controls")

        algo_choice = st.selectbox(
            "Select Algorithm to Tune",
            ["K-Nearest Neighbors (KNN)", "Decision Tree Classifier", "Logistic Regression"]
        )

        # Configure model parameters dynamically
        if algo_choice == "K-Nearest Neighbors (KNN)":
            st.markdown("**KNN Classifier Parameters:**")
            k_val = st.slider("Neighborhood Size (K)", min_value=1, max_value=15, value=3, step=1)
            clf = KNeighborsClassifier(n_neighbors=k_val)
            params_desc = f"n_neighbors={k_val}"

        elif algo_choice == "Decision Tree Classifier":
            st.markdown("**Decision Tree Parameters:**")
            depth_val = st.slider("Maximum Depth Limit (max_depth)", min_value=1, max_value=10, value=3, step=1)
            clf = DecisionTreeClassifier(max_depth=depth_val, random_state=42)
            params_desc = f"max_depth={depth_val}"

        elif algo_choice == "Logistic Regression":
            st.markdown("**Logistic Regression Parameters:**")
            c_val = st.select_slider(
                "Inverse Regularization Strength (C)",
                options=[0.01, 0.1, 1.0, 10.0, 100.0],
                value=1.0
            )
            clf = LogisticRegression(C=c_val, max_iter=200, random_state=42)
            params_desc = f"C={c_val}, max_iter=200"

        # Fit model
        clf.fit(X_train, y_train)
        y_train_pred = clf.predict(X_train)
        y_test_pred = clf.predict(X_test)

        # Evaluate scores
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)

        # Display accuracies
        st.markdown("---")
        st.markdown("**Current Model Metrics:**")
        m1, m2 = st.columns(2)
        m1.metric("Training Accuracy", f"{train_acc * 100:.2f}%")
        delta = test_acc - train_acc
        m2.metric(
            "Validation Accuracy",
            f"{test_acc * 100:.2f}%",
            delta=f"{delta * 100:.2f}% variance",
            delta_color="inverse" if delta < -0.05 else "normal"
        )

        # Diagnosing model state
        if train_acc == 1.0 and test_acc < 0.95:
            st.warning("⚠️ **Overfitting Detected**: The model fits the training split perfectly but exhibits a drop in generalizing validation data.")
        elif train_acc < 0.85:
            st.error("⚠️ **Underfitting Detected**: Model representation complexity is too low. Performance on training data is suboptimal.")
        else:
            st.success("✅ **Balanced Generalization**: Optimal performance across training and validation splits.")

        st.markdown("</div>", unsafe_allow_html=True)

    with col_results:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown(f"#### 📊 Performance Visualizations ({algo_choice})")

        # Plot Confusion Matrix
        fig_cm, ax_cm = plt.subplots(figsize=(4.5, 3.2))
        fig_cm.patch.set_facecolor('#0f172a')
        ax_cm.set_facecolor('#1e293b')
        cm = confusion_matrix(y_test, y_test_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names, ax=ax_cm)
        ax_cm.tick_params(colors='white')
        ax_cm.xaxis.label.set_color('white')
        ax_cm.yaxis.label.set_color('white')
        for text in ax_cm.texts:
            text.set_color('white')
        plt.xlabel('Predicted Label', color='white')
        plt.ylabel('True Label', color='white')
        plt.title("Confusion Matrix (Validation Data)", color='white')

        col_cm, col_report = st.columns([1, 1.2])
        with col_cm:
            st.pyplot(fig_cm)
        with col_report:
            st.write("**Model Classification Report:**")
            report = classification_report(y_test, y_test_pred, target_names=iris.target_names, output_dict=True)
            report_df = pd.DataFrame(report).transpose().round(2)
            st.dataframe(report_df.iloc[:-3, :-1], use_container_width=True)
            st.caption("Precision measures classification exactness; Recall measures classification completeness.")

        st.markdown("</div>", unsafe_allow_html=True)

    # Bottom section: Comparative Benchmarking Summary
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("#### 📋 System Benchmark Comparative Analysis")
    st.write("Cross-algorithm benchmarking using default settings on a stratified 80/20 partition split:")

    # Train default versions of all three models
    knn_def = KNeighborsClassifier(n_neighbors=3).fit(X_train, y_train)
    dt_def = DecisionTreeClassifier(max_depth=3, random_state=42).fit(X_train, y_train)
    lr_def = LogisticRegression(C=1.0, max_iter=200, random_state=42).fit(X_train, y_train)

    summary_data = {
        "Classifier Model": ["Logistic Regression (Parametric)", "K-Nearest Neighbors (Instance)", "Decision Tree (Rule-based)"],
        "Hyperparameters": ["C=1.0, penalty=L2", "K=3, metric=Euclidean", "max_depth=3, split=Gini"],
        "Training Accuracy": [
            f"{accuracy_score(y_train, lr_def.predict(X_train))*100:.2f}%",
            f"{accuracy_score(y_train, knn_def.predict(X_train))*100:.2f}%",
            f"{accuracy_score(y_train, dt_def.predict(X_train))*100:.2f}%"
        ],
        "Validation Accuracy": [
            f"{accuracy_score(y_test, lr_def.predict(X_test))*100:.2f}%",
            f"{accuracy_score(y_test, knn_def.predict(X_test))*100:.2f}%",
            f"{accuracy_score(y_test, dt_def.predict(X_test))*100:.2f}%"
        ]
    }
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ==============================================================================
# TAB 3: REAL-TIME INFERENCE ENGINE
# ==============================================================================
with tab3:
    st.markdown("<div class='section-title'>Real-time Classification Inference Engine</div>", unsafe_allow_html=True)

    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.write(
        f"This panel executes inference tasks using the trained **model parameters optimized in the Benchmarking tab** "
        f"(`{algo_choice}` Classifier)."
    )

    # Morphological parameter inputs
    p_col1, p_col2, p_col3, p_col4 = st.columns(4)
    with p_col1:
        s_length = st.slider("Sepal Length (cm)", min_value=4.0, max_value=8.0, value=5.8, step=0.1)
    with p_col2:
        s_width = st.slider("Sepal Width (cm)", min_value=2.0, max_value=4.5, value=3.0, step=0.1)
    with p_col3:
        p_length = st.slider("Petal Length (cm)", min_value=1.0, max_value=7.0, value=4.35, step=0.1)
    with p_col4:
        p_width = st.slider("Petal Width (cm)", min_value=0.1, max_value=2.5, value=1.3, step=0.1)

    # Input vector formatted as pandas DataFrame to prevent feature name warnings
    features = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
    input_sample = pd.DataFrame([[s_length, s_width, p_length, p_width]], columns=features)

    # Predict
    pred_species_id = clf.predict(input_sample)[0]
    pred_name = iris.target_names[pred_species_id]

    # Probability estimation
    if hasattr(clf, "predict_proba"):
        pred_probs = clf.predict_proba(input_sample)[0]
    else:
        # Fallback representation for models without native probabilities
        pred_probs = [0.0, 0.0, 0.0]
        pred_probs[pred_species_id] = 1.0

    badge_class = f"badge-{pred_name}"

    pred_col1, pred_col2 = st.columns([1, 2.2])
    with pred_col1:
        st.markdown(f"""
        <div style='text-align: center; padding: 1.5rem; border-radius: 12px; background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.05); margin-top: 1rem;'>
            <h5 style='margin-top: 0; margin-bottom: 1rem;'>System Classification Outcome:</h5>
            <span class='badge {badge_class}' style='font-size: 1.35rem; padding: 0.6rem 1.75rem; box-shadow: 0 4px 15px rgba(0,0,0,0.2);'>{pred_name}</span>
        </div>
        """, unsafe_allow_html=True)

    with pred_col2:
        st.markdown("<h5 style='margin-top: 1rem;'>Confidence Probabilities:</h5>", unsafe_allow_html=True)
        for i, name in enumerate(iris.target_names):
            prob = pred_probs[i]
            st.write(f"**{name.capitalize()}**: {prob*100:.1f}%")
            st.progress(float(prob))

    st.markdown("</div>", unsafe_allow_html=True)

# ==============================================================================
# TAB 4: SYSTEM ARCHITECTURE & MATH
# ==============================================================================
with tab4:
    st.markdown("<div class='section-title'>System Design & Algorithm Blueprint</div>", unsafe_allow_html=True)

    col_arch1, col_arch2 = st.columns(2)

    with col_arch1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("#### 📐 Mathematical Formulations")

        st.markdown("##### 1. Logistic Regression Model")
        st.write(
            "Logistic Regression maps input dimensions $X$ to species class probabilities using the **Softmax function**:"
        )
        st.latex(r"P(y = c \mid X) = \frac{e^{w_c^T X + b_c}}{\sum_{j=1}^{C} e^{w_j^T X + b_j}}")
        st.write("Where $w$ and $b$ represent weights and biases learned via minimizing cross-entropy loss with an L2 regularization penalty.")

        st.markdown("##### 2. K-Nearest Neighbors Classifier")
        st.write(
            "KNN assigns a class membership to a query point based on a distance metric. Our system uses the **Euclidean Distance** in a 4-dimensional feature space:"
        )
        st.latex(r"d(p, q) = \sqrt{\sum_{i=1}^{4} (p_i - q_i)^2}")
        st.write(r"The classification decision is modeled as a simple majority vote: $y = \text{mode}(y_{i_1}, \dots, y_{i_K})$.")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_arch2:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("##### 3. Decision Tree Splitting Heuristics")
        st.write(
            "The Decision Tree partitions the feature space recursively. At each node, the feature and threshold are chosen to maximize "
            "information gain, minimizing the **Gini Impurity index**:"
        )
        st.latex(r"I_G(p) = 1 - \sum_{i=1}^{C} p_i^2")
        st.write("Where $p_i$ is the probability of a sample belonging to class $i$ at the target node.")

        st.markdown("---")
        st.markdown("#### 📂 Modular Pipeline Structure")
        st.write("The software design utilizes a clean separation of concerns:")
        st.markdown(
            "1. **Data ingestion module** (`load_iris` extraction and parser)\n"
            "2. **Visualization engine** (matplotlib/seaborn generator plotting correlation matrices and distribution reports)\n"
            "3. **Training & validation module** (scikit-learn split models and parameter evaluation scripts)\n"
            "4. **Inference engine** (takes raw morphological vectors and computes predicted species indices and probability arrays)"
        )
        st.markdown("</div>", unsafe_allow_html=True)
