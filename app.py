import streamlit as st
import pandas as pd
import joblib
import numpy as np
from pathlib import Path

############################################################
# Page & Style Config                                      #
############################################################

st.set_page_config(
    page_title="Disease Prediction App",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -- Custom CSS ---------------------------------------------------------
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-bottom: 1rem;
    }
    .disclaimer {
        font-size: 0.8rem;
        color: #757575;
        font-style: italic;
        margin-top: 2rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

# -- Title & Description ------------------------------------------------
st.markdown(
    "<h1 class='main-header'>Disease Prediction System</h1>", unsafe_allow_html=True
)
st.markdown(
    "<p class='sub-header'>Select your symptoms to get a prediction</p>",
    unsafe_allow_html=True,
)

############################################################
# Data Loaders                                             #
############################################################


@st.cache_resource(show_spinner=False)
def load_model_features_mapping():
    """Return trained model, feature list and disease→specialty mapping"""
    model_path = Path("model/disease_prediction_multinomial_naive_bayes.joblib")
    feature_path = Path("model/model_features.csv")
    mapping_path = Path("model/disease_specialty_mapping.tsv")

    # ---- Model
    model = joblib.load(model_path)

    # ---- Feature List
    if feature_path.exists():
        features = pd.read_csv(feature_path)["feature"].tolist()
    else:
        # Fallback to model.coef_ shape if csv missing (keeps previous behaviour)
        features = [f"symptom_{i}" for i in range(model.coef_.shape[1])]

    # ---- Disease→Specialty Mapping
    if mapping_path.exists():
        mapping_df = pd.read_csv(mapping_path, sep="\t").fillna("")
        mapping = mapping_df.set_index("Disease")[
            ["Specialization", "Classification", "Taxonomy Code"]
        ].to_dict(orient="index")
    else:
        mapping = {}
        st.warning("Specialty mapping file not found – specialties will be omitted.")

    return model, features, mapping


model, all_symptoms, disease_to_specialty = load_model_features_mapping()

############################################################
# Helper Functions                                         #
############################################################


def one_hot(symptoms, all_feats):
    """Return DataFrame with one‑hot encoded symptoms."""
    df = pd.DataFrame(0, index=[0], columns=all_feats)
    df.loc[0, symptoms] = 1
    return df


def render_specialty_block(disease: str):
    """Show recommended specialty (if mapping available)."""
    info = disease_to_specialty.get(disease.lower()) or disease_to_specialty.get(
        disease
    )
    if not info:
        st.info("No specialty mapping available for this disease.")
        return

    spec = info.get("Specialization") or info.get("Classification")
    taxonomy = info.get("Taxonomy Code")

    st.markdown("### Recommended Specialty")
    st.success(f"**{spec}**  ")
    if taxonomy:
        st.caption(f"NPI Taxonomy Code: {taxonomy}")


############################################################
# Main Layout                                              #
############################################################

col_left, col_right = st.columns([1, 1])

# -- Symptoms Input ---------------------------------------
with col_left:
    st.markdown("### Select Your Symptoms")
    selected_symptoms = st.multiselect(
        "Start typing to search…", options=sorted(all_symptoms), default=[]
    )

    st.write(f"Selected: **{len(selected_symptoms)}** symptom(s)")

    predict_btn = st.button("Predict Disease", disabled=not selected_symptoms)

# -- Prediction Output ------------------------------------
with col_right:
    st.markdown("### Prediction Results")

    if predict_btn and selected_symptoms:
        try:
            X = one_hot(selected_symptoms, all_symptoms)
            with st.spinner("Analyzing symptoms…"):
                y_pred = model.predict(X)[0]
                st.markdown(f"## Likely Disease: **{y_pred}**")

                # Show top‑3 probabilities (if available)
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(X)[0]
                    top_idx = np.argsort(probs)[-3:][::-1]
                    st.markdown("### Other possibilities")
                    for i in top_idx:
                        if probs[i] < 0.05:
                            continue
                        st.markdown(f"• {model.classes_[i]} — {probs[i]*100:.1f}%")

            # ---- Specialty block
            render_specialty_block(y_pred)

            # ---- Store in session for downstream use
            st.session_state["last_prediction"] = y_pred

        except Exception as e:
            st.error(f"Prediction failed: {e}")

############################################################
# Sidebar Information                                      #
############################################################
with st.sidebar:
    st.markdown("<h1 style='text-align: center'>🏥</h1>", unsafe_allow_html=True)
    st.title("About This App")
    st.info(
        "This application uses a trained machine‑learning model to predict "
        "possible diseases from selected symptoms. Results are for educational "
        "purposes only and not a substitute for professional medical advice."
    )

    st.markdown("### How to use:")
    st.markdown(
        """
        1. Search & select your symptoms (multi‑select).  
        2. Click **Predict Disease** to run the model.  
        3. Review the likely disease, alternative possibilities, and suggested specialty.  
        4. Consult a qualified healthcare professional for clinical concerns.
        """
    )

    st.markdown("### Model Information:")
    st.markdown(
        f"""- **Algorithm**: Multinomial Naïve Bayes  
- **Number of symptoms**: {len(all_symptoms)}  
- **Specialty mapping**: {'✔️' if disease_to_specialty else '❌'}"""
    )

############################################################
# Footer                                                   #
############################################################

st.markdown(
    """<div class='disclaimer'>Disclaimer: This tool is for educational purposes only. It is not a substitute for medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider with questions about a medical condition.</div>""",
    unsafe_allow_html=True,
)
