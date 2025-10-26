
# ================================
# 🏡 Ames Housing Price Predictor
# ================================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import traceback
import matplotlib.pyplot as plt
from typing import Dict, List

from source.feature_engineer import FeatureEngineer
from source.log_transformer import LogTransformer

# Optional SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

# sklearn helpers
try:
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
except Exception:
    OneHotEncoder = None
    ColumnTransformer = None
    Pipeline = None

# -------------------------
# Streamlit Page config
# -------------------------

st.set_page_config(page_title="Ames Housing Price Predictor", page_icon="🏡", layout="wide")
st.title("🏡 Ames Housing Price Predictor")

# -------------------------
# Project Introduction
# -------------------------
st.markdown("""
### 🌟 Introduction  
This intelligent predictive system estimates the **sale price of residential homes in Ames, Iowa** using advanced machine learning.  
The model powering this app is a highly optimized **Gradient Boosting Regression Model** — selected as the **best performer** after comparing  around **20 different regression models** including Elastic Net Regression, Lasso Regression, Ridge Regression, Polynomial Regression, Stochastic Gradient Regression, Batch Gradient Regression, Linear Regression,Linear  SVR, Polynomial SVR, RBF SVR, Decision Tree, Voting Regressor RMSE, Bagging , Random Forest Regressor, Extra Tree Regressor, ADA Boost, Gradient Boosting, XGBRegressor, LGBMRegressor and CatBoostRegressor.

After comprehensive **hyperparameter tuning** and **cross-validation**, Gradient Boosting achieved the **lowest RMSE** while maintaining robust generalization and interpretability.  
Its ability to model complex nonlinear relationships and feature interactions made it ideal for predicting property values where many attributes (size, quality, year, neighborhood, etc.) interact in subtle ways.

This app integrates that final trained model into a scikit-learn **pipeline** with preprocessing, feature engineering, and transformation — delivering **real-time, explainable predictions** through an interactive Streamlit interface.
""")

st.markdown("---")

st.markdown(
    "Use the sidebar to enter property details. Click **Predict Sale Price 💰** to see your prediction and SHAP-based explanations showing how each feature influenced the estimate."
)


# -------------------------
# Load pipeline + feature metadata
# -------------------------
@st.cache_resource
def load_artifacts():
    errors = []
    pipe = None
    feature_names = None
    try:
        pipe = joblib.load("models/final_pipeline.pkl")
    except Exception as e:
        errors.append(f"Could not load 'final_pipeline.pkl': {e}")
    try:
        feature_names = joblib.load("models/feature_names.pkl")
    except Exception as e:
        errors.append(f"Could not load 'feature_names.pkl': {e}")
    return pipe, feature_names, errors

pipe, feature_names, load_errors = load_artifacts()
if load_errors:
    for e in load_errors:
        st.error(e)
    st.stop()
if pipe is None:
    st.error("Pipeline not found. Make sure 'final_pipeline.pkl' is in the same directory.")
    st.stop()

preprocessor = None
model = None
try:
    preprocessor = pipe.named_steps.get("preprocessor", None)
    model = pipe.named_steps.get("model", None)
except Exception:
    try:
        if isinstance(pipe, Pipeline):
            steps = pipe.steps
            if len(steps) >= 1:
                model = steps[-1][1]
                for name, obj in steps[:-1]:
                    if ColumnTransformer is not None and isinstance(obj, ColumnTransformer):
                        preprocessor = obj
                        break
    except Exception:
        pass

# -------------------------
# Helper to extract categorical info
# -------------------------
import inspect
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

def extract_categorical_info(preprocessor):
    info = {}
    if not hasattr(preprocessor, "transformers_"):
        return info
    for name, transformer, cols in preprocessor.transformers_:
        if hasattr(transformer, "named_steps"):
            if name == "nom":
                encoder = transformer.named_steps.get("onehot")
            elif name == "ord":
                encoder = transformer.named_steps.get("encoder")
            else:
                continue
            if encoder and hasattr(encoder, "categories_"):
                for col, cats in zip(cols, encoder.categories_):
                    info[col] = list(cats)
    return info

# -------------------------
# Sidebar input setup
# -------------------------
st.sidebar.header("🏠 Input House Features")

defaults = {
    "LotArea": 8000,
    "GrLivArea": 1500,
    "GarageCars": 2,
    "OverallQual": 5,
    "YearRemodAdd": 2000,
    "TotalBsmtSF": 800,
    "1stFlrSF": 800,
    "TotalBath": 2.0,
    "BsmtFinSF1": 300,
    "GarageArea": 350,
}

sidebar_values = {}
# Numeric inputs
sidebar_values["LotArea"] = st.sidebar.number_input("LotArea (sq ft)", 0, 100000, int(defaults["LotArea"]), step=100)
sidebar_values["GrLivArea"] = st.sidebar.number_input("GrLivArea (sq ft)", 0, 10000, int(defaults["GrLivArea"]), step=10)
sidebar_values["TotalBsmtSF"] = st.sidebar.number_input("TotalBsmtSF (sq ft)", 0, 4000, int(defaults["TotalBsmtSF"]), step=10)
sidebar_values["1stFlrSF"] = st.sidebar.number_input("1stFlrSF (sq ft)", 0, 4000, int(defaults["1stFlrSF"]), step=10)
sidebar_values["BsmtFinSF1"] = st.sidebar.number_input("BsmtFinSF1", 0, 2000, int(defaults["BsmtFinSF1"]), step=10)
sidebar_values["GarageCars"] = st.sidebar.slider("GarageCars", 0, 5, int(defaults["GarageCars"]))
sidebar_values["GarageArea"] = st.sidebar.number_input("GarageArea (sq ft)", 0, 2000, int(defaults["GarageArea"]), step=10)
sidebar_values["OverallQual"] = st.sidebar.slider("OverallQual (1–10)", 1, 10, int(defaults["OverallQual"]))
sidebar_values["YearRemodAdd"] = st.sidebar.number_input("YearRemodAdd", 1870, 2025, int(defaults["YearRemodAdd"]))
sidebar_values["TotalBath"] = st.sidebar.slider("TotalBath", 0.0, 10.0, defaults["TotalBath"], step=0.5)

# Categorical dropdowns
cat_info = extract_categorical_info(preprocessor)
categorical_cols = list(cat_info.keys())
for feat in categorical_cols:
    cats = cat_info.get(feat, ["TA"])
    default_index = cats.index("TA") if "TA" in cats else 0
    sidebar_values[feat] = st.sidebar.selectbox(feat, options=cats, index=default_index)

# -------------------------
# Prepare input dataframe
# -------------------------
if feature_names is None:
    st.error("feature_names.pkl not found.")
    st.stop()

input_df = pd.DataFrame(columns=feature_names)
input_df.loc[0] = [np.nan] * len(feature_names)

# Fill known columns
for k, v in sidebar_values.items():
    if k in input_df.columns:
        input_df.at[0, k] = v
    else:
        matches = [c for c in input_df.columns if k.lower() in c.lower()]
        if matches:
            input_df.at[0, matches[0]] = v

# Derive bath-related cols
if "TotalBath" in sidebar_values:
    total = float(sidebar_values["TotalBath"])
    full = int(total // 1)
    half = int((total - full) * 2)
    for bcol, val in [("FullBath", full), ("HalfBath", half)]:
        if bcol in input_df.columns:
            input_df.at[0, bcol] = val

# Default fill for categorical
DEFAULT_CATEGORY = {"KitchenQual": "TA", "ExterQual": "TA", "BsmtQual": "TA", "FireplaceQu": "TA", "GarageFinish": "Unf"}
for c in categorical_cols:
    if c in input_df.columns and (pd.isna(input_df.at[0, c]) or input_df.at[0, c] == ""):
        cats = cat_info.get(c, [])
        input_df.at[0, c] = "TA" if "TA" in cats else DEFAULT_CATEGORY.get(c, "TA")

# Convert numeric & categorical dtypes
for col in input_df.columns:
    if col in categorical_cols:
        input_df[col] = input_df[col].astype(str)
    else:
        input_df[col] = pd.to_numeric(input_df[col], errors="coerce")

st.subheader("🧾 Input preview")
st.dataframe(input_df.fillna("").rename(columns={0: "value"}))

# -------------------------
# Prediction + SHAP explainability
# -------------------------
if st.button("💰 Predict Sale Price"):
    try:
        preds = pipe.predict(input_df)
        raw_pred = float(preds[0])
        predicted_price = float(np.expm1(raw_pred)) if raw_pred < 20 else raw_pred
        predicted_price = max(predicted_price, 0.0)
        st.success(f"🏠 Predicted Sale Price: **${predicted_price:,.0f}**")
        st.caption(f"Model raw output: `{raw_pred:.6f}`")

        # --- SHAP EXPLANATION ---
        if SHAP_AVAILABLE and model is not None:
            st.markdown("---")
            st.subheader("🔍 Model Explainability with SHAP")

            try:
                if preprocessor is not None:
                    X_trans = preprocessor.transform(input_df)
                    try:
                        feature_names_trans = preprocessor.get_feature_names_out()
                    except Exception:
                        feature_names_trans = list(input_df.columns)
                else:
                    X_trans = input_df.values
                    feature_names_trans = list(input_df.columns)

                model_name = type(model).__name__
                if any(x in model_name.lower() for x in ["forest", "boost", "tree", "xgb", "lgbm"]):
                    explainer = shap.TreeExplainer(model)
                else:
                    explainer = shap.Explainer(model, X_trans)

                shap_values = explainer(X_trans)

                if hasattr(shap_values, "values") and np.allclose(shap_values.values, 0):
                    st.warning("SHAP values are all zero — model output might be constant.")
                else:
                    # =========================
                    # 📊 SHAP Summary Plot
                    # =========================
                    st.markdown("## 📊 SHAP Summary Plot – Global Feature Importance")
                    colA, colB = st.columns([1, 1.6])
                    with colA:
                        st.markdown(
                            """
                            **What it shows:**  
                            - Average influence of each feature on predicted house prices.  
                            - Longer bars → greater impact.  
                            - Positive = higher predicted price, negative = lower.  
                            **Example:** `GrLivArea` and `OverallQual` are usually strong positive drivers.
                            """
                        )
                    with colB:
                        fig_s = plt.figure(figsize=(10, 5))
                        shap.summary_plot(
                            shap_values.values if hasattr(shap_values, "values") else shap_values,
                            features=X_trans,
                            plot_type="bar",
                            feature_names=feature_names_trans,
                            show=False,
                        )
                        st.pyplot(fig_s)
                        plt.close(fig_s)

                    st.markdown("---")

                    # =========================
                    # 💧 SHAP Waterfall Plot
                    # =========================
                    st.markdown("## 💧 SHAP Waterfall Plot – Individual Prediction Explanation")
                    colC, colD = st.columns([1, 1.6])
                    with colC:
                        st.markdown(
                            """
                            **How this works:**  
                            - Explains **this single prediction**.  
                            - Gray line = model’s average output E[f(X)].  
                            - 🔴 Red bars increase price, 🔵 Blue bars decrease it.  
                            - Rightmost value = final predicted price f(x). 
                            **Example:** High `OverallQual` may push price upward; low `LotArea` may pull it down.
                            """
                        )
                    with colD:
                        fig_w = plt.figure(figsize=(10, 6))
                        shap.plots.waterfall(
                            shap.Explanation(
                                values=shap_values[0].values,
                                base_values=shap_values[0].base_values,
                                data=shap_values[0].data,
                                feature_names=feature_names_trans,
                            ),
                            show=False,
                        )
                        st.pyplot(fig_w)
                        plt.close(fig_w)

            except Exception as e:
                st.warning(f"SHAP could not compute values: {e}")

    except Exception:
        st.error("❌ Prediction failed:")
        st.text(traceback.format_exc())


# -------------------------
# 📘 Project Overview (Footer)
# -------------------------
st.markdown("---")
st.markdown("""
### 🧩 About This Project  
This interactive web app is the culmination of a complete **end-to-end data science and machine learning pipeline** developed for the **Ames Housing Price Prediction Project** — a regression-based model designed to estimate real estate prices in Ames, Iowa.

---

### 🧱 **1. Data Understanding and Preprocessing**  
The process began with **Exploratory Data Analysis (EDA)** to understand feature distributions, detect missing values, and visualize relationships within the dataset.  
- Constructed **correlation matrices** and **heatmaps** to identify the strongest predictors of `SalePrice` — including `OverallQual`, `GrLivArea`, and `TotalSF`.  
- Used **pairplots and boxplots** to study how numerical and categorical features influence the target.  
- Detected strong positive correlations among floor areas and quality-related variables, and negative impacts from age-related and poor-condition variables.  
- Addressed missing data with context-specific imputations (e.g., replacing missing categorical entries with `"None"` or `"TA"` and numerical ones with median values).

---

### ⚙️ **2. Feature Engineering and Transformation**  
To improve predictive performance and capture non-linear relationships:  
- Created composite features such as **TotalSF**, **TotalBath**, and **TotalPorchSF** to represent combined living areas, total bathrooms, and porch spaces.  
- Dropped redundant or low-impact columns identified during EDA.  
- Applied a **logarithmic transformation** (`log1p`) to the target variable `SalePrice` to normalize its right-skewed distribution, stabilizing variance and improving model fit.  
- Log-transformed several skewed predictors to make relationships more linear.  
- Performed **encoding** of categorical variables using **OrdinalEncoder** for ordered features and **OneHotEncoder** for nominal ones.  
- Implemented scaling and standardization in the preprocessing pipeline for consistent feature magnitudes.

---

### 🧮 **3. Data Visualization and Dimensionality Reduction**  
To explore global data patterns, **Principal Component Analysis (PCA)** was performed on standardized features.  
- PCA helped visualize variance structure in 2D and 3D projections, revealing clear clustering of homes based on quality and size.  
- Learned that only a handful of engineered and quality features capture most of the variance, confirming the dataset’s strong feature redundancy.  
- These visual insights guided feature selection and supported simplifying the model without losing predictive power.

---

### 🤖 **4. Model Building and Evaluation**  
An extensive **model benchmarking** process was conducted, testing around **20 different algorithms** including:  
- **Linear Models:** Linear Regression (Normal Equation), Ridge, Lasso, ElasticNet  
- **Tree-Based Models:** Decision Tree, Random Forest, Extra Trees, AdaBoost, Gradient Boosting, XGBoost, LightGBM, CatBoost  
- **Others:** Polynomial Regression, Batch Gradient Descent (BGD), Stochastic Gradient Descent (SGD), Linear SVR,     Polynomial SVR, RBF SVR, Voting Regressor, Bagging, and other ensemble combinations.  

Each model was evaluated using **Root Mean Squared Error (RMSE)** through cross-validation and train–test splits.  
- Learning curves and residual plots were analyzed to check for overfitting or underfitting.  
- Regularization methods (Ridge/Lasso) helped identify feature importance and reduce multicollinearity.  
- Ensemble methods (RandomForest, XGBoost, GradientBoosting) provided the lowest RMSE scores and superior generalization.

---
""")

# =========================
# 🧮 Model Evaluation Table + Learning Curve
# =========================
# =========================
# 🧮 Model Evaluation Table + Learning Curve
# =========================
st.markdown("### 🧮 Model Selection and Evaluation")

# Create two columns — table (left) and image (right)
col1, col2 = st.columns([1.3, 1],vertical_alignment="center")

with col1:
    st.markdown("""
    During model experimentation, around **20 regression algorithms** were trained and evaluated using
    **cross-validated Root Mean Squared Error (RMSE)** and **learning curve analysis**.

    <div style='text-align: center'>
    <table style='margin: 0 auto; border-collapse: collapse;'>
        <tr style='background-color:#2d1ac4; font-weight:bold;'>
            <th style='padding: 8px 20px; border: 1px solid #ccc;'>Model</th>
            <th style='padding: 8px 20px; border: 1px solid #ccc;'>RMSE</th>
        </tr>
        <tr><td>Elastic Net Regression</td><td>0.1313</td></tr>
        <tr><td>Lasso Regression</td><td>0.1317</td></tr>
        <tr><td>Ridge Regression</td><td>0.1343</td></tr>
        <tr><td>Polynomial Regression</td><td>6.5049</td></tr>
        <tr><td>Stochastic Gradient Regression</td><td>1.1073</td></tr>
        <tr><td>Batch Gradient Regression</td><td>1.1073</td></tr>
        <tr><td>Linear SVR</td><td>1.1072</td></tr>
        <tr><td>Poly SVR</td><td>0.1309</td></tr>
        <tr><td>RBF SVR</td><td>0.1290</td></tr>
        <tr><td>Decision Tree</td><td>0.1888</td></tr>
        <tr><td>Voting Regressor</td><td>0.1278</td></tr>
        <tr><td>Bagging Regressor</td><td>0.1460</td></tr>
        <tr><td>Random Forest Regressor</td><td>0.1380</td></tr>
        <tr><td>Extra Trees Regressor</td><td>0.1426</td></tr>
        <tr><td>AdaBoost Regressor</td><td>0.1432</td></tr>
        <tr style='background-color:#FDEDEC; font-weight:bold; color:#C0392B;'>
            <td>Gradient Boosting Regressor</td><td>🎯 <b>0.1247 (Best)</b></td>
        </tr>
        <tr><td>XGBoost Regressor</td><td>0.1364</td></tr>
        <tr><td>LightGBM Regressor</td><td>0.1316</td></tr>
        <tr><td>CatBoost Regressor</td><td>0.1276</td></tr>
    </table>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(
    """


    
    """,
    unsafe_allow_html=True,
    )
    try:
        st.image("images/gradient_boosting_learningcurve.png", caption="Gradient Boosting Learning Curve", width=500)
    except Exception:
        st.warning("⚠️ Learning curve image not found. Please ensure 'gradient_boosting_learningcurve.png' is in the app directory.")
        st.markdown("", unsafe_allow_html=True)

# Add explanation below both columns
st.markdown(""" 
## 🏆 RMSE as the Primary Indicator

The most critical factor for selection is the **Root Mean Square Error (RMSE)** from the comparison table.

* The Gradient Boosting Regressor achieved the **lowest RMSE of 0.1247** among all 18 models tested.
* Since RMSE measures the average magnitude of the errors, the lowest value signifies that this model produces the **most accurate predictions** and is therefore the superior choice for the project.

## 📈 Learning Curve Demonstrating Stability

The learning curve confirms the model's robustness and generalization ability, despite a slight tendency toward overfitting (the gap between the training and validation curves):

* **Validation Error is Low:** The Validation RMSE curve stabilizes at a relatively **low value (around 0.13)**. This low error confirms the model is generalizing well to unseen data, even as the training error remains significantly lower.
* **Sufficient Data:** Both the Training and Validation RMSE curves have **converged** (flattened out) after a training set size of about 600-800. This indicates that the model has learned all it can from the current dataset and its performance is **stable**, suggesting it's a reliable final choice.

The Gradient Boosting Regressor was chosen because it delivered the **minimum prediction error (RMSE)** while maintaining **stable and competitive performance** on new data, as confirmed by its learning curve.
""")

st.markdown("""
---

### 💡 **6. Interpretability and Insights**  
To ensure model transparency, **SHAP (SHapley Additive exPlanations)** was applied:  
- **Global SHAP plots** highlight the most influential features overall (e.g., `GrLivArea`, `OverallQual`, `TotalSF`, `GarageCars`).  
- **Local SHAP waterfall plots** explain individual predictions by showing how each feature pushes the prediction higher or lower than the model’s baseline.  
This allows users to not only get a price estimate but also *understand why* a home’s predicted value is high or low.

---

### 🏁 **7. Key Learnings and Takeaways**  
Through this project, I learned how data preprocessing quality directly impacts model accuracy, how feature engineering can outperform complex tuning, and how interpretability tools like SHAP bridge the gap between model performance and trust.  
The combination of **EDA, transformation, dimensional analysis, and explainability** led to a highly interpretable and accurate regression model — now deployed interactively using Streamlit for real-world usability.
""")
