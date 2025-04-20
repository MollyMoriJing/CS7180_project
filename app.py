import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="Disease Prediction App",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Add custom CSS
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

# Title and description
st.markdown(
    "<h1 class='main-header'>Disease Prediction System</h1>", unsafe_allow_html=True
)
st.markdown(
    "<p class='sub-header'>Select your symptoms to get a prediction</p>",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model_and_features():
    """Load the model and features list (cached to avoid reloading)"""
    model = joblib.load("model/disease_prediction_multinomial_naive_bayes.joblib")

    # Try to load features from CSV if available
    try:
        features = pd.read_csv("model/model_features.csv")["feature"].tolist()
    except:
        # If CSV isn't available, use the symptoms from your list
        features = [
            "anxiety and nervousness",
            "depression",
            "shortness of breath",
            "depressive or psychotic symptoms",
            "sharp chest pain",
            "dizziness",
            "insomnia",
            "abnormal involuntary movements",
            "chest tightness",
            "palpitations",
            "irregular heartbeat",
            "breathing fast",
            "hoarse voice",
            "sore throat",
            "difficulty speaking",
            "cough",
            "nasal congestion",
            "throat swelling",
            "diminished hearing",
            "lump in throat",
            "throat feels tight",
            "difficulty in swallowing",
            "skin swelling",
            "retention of urine",
            "groin mass",
            "leg pain",
            "hip pain",
            "suprapubic pain",
            "blood in stool",
            "lack of growth",
            "emotional symptoms",
            "elbow weakness",
            "back weakness",
            "pus in sputum",
            "symptoms of the scrotum and testes",
            "swelling of scrotum",
            "pain in testicles",
            "flatulence",
            "pus draining from ear",
            "jaundice",
            "mass in scrotum",
            "white discharge from eye",
            "irritable infant",
            "abusing alcohol",
            "fainting",
            "hostile behavior",
            "drug abuse",
            "sharp abdominal pain",
            "feeling ill",
            "vomiting",
            "headache",
            "nausea",
            "diarrhea",
            "vaginal itching",
            "vaginal dryness",
            "painful urination",
            "involuntary urination",
            "pain during intercourse",
            "frequent urination",
            "lower abdominal pain",
            "vaginal discharge",
            "blood in urine",
            "hot flashes",
            "intermenstrual bleeding",
            "hand or finger pain",
            "wrist pain",
            "hand or finger swelling",
            "arm pain",
            "wrist swelling",
            "arm stiffness or tightness",
            "arm swelling",
            "hand or finger stiffness or tightness",
            "wrist stiffness or tightness",
            "lip swelling",
            "toothache",
            "abnormal appearing skin",
            "skin lesion",
            "acne or pimples",
            "dry lips",
            "facial pain",
            "mouth ulcer",
            "skin growth",
            "eye deviation",
            "diminished vision",
            "double vision",
            "cross-eyed",
            "symptoms of eye",
            "pain in eye",
            "eye moves abnormally",
            "abnormal movement of eyelid",
            "foreign body sensation in eye",
            "irregular appearing scalp",
            "swollen lymph nodes",
            "back pain",
            "neck pain",
            "low back pain",
            "pain of the anus",
            "pain during pregnancy",
            "pelvic pain",
            "impotence",
            "infant spitting up",
            "vomiting blood",
            "regurgitation",
            "burning abdominal pain",
            "restlessness",
            "symptoms of infants",
            "wheezing",
            "peripheral edema",
            "neck mass",
            "ear pain",
            "jaw swelling",
            "mouth dryness",
            "neck swelling",
            "knee pain",
            "foot or toe pain",
            "bowlegged or knock-kneed",
            "ankle pain",
            "bones are painful",
            "knee weakness",
            "elbow pain",
            "knee swelling",
            "skin moles",
            "knee lump or mass",
            "weight gain",
            "problems with movement",
            "knee stiffness or tightness",
            "leg swelling",
            "foot or toe swelling",
            "heartburn",
            "smoking problems",
            "muscle pain",
            "infant feeding problem",
            "recent weight loss",
            "problems with shape or size of breast",
            "underweight",
            "difficulty eating",
            "scanty menstrual flow",
            "vaginal pain",
            "vaginal redness",
            "vulvar irritation",
            "weakness",
            "decreased heart rate",
            "increased heart rate",
            "bleeding or discharge from nipple",
            "ringing in ear",
            "plugged feeling in ear",
            "itchy ear(s)",
            "frontal headache",
            "fluid in ear",
            "neck stiffness or tightness",
            "spots or clouds in vision",
            "eye redness",
            "lacrimation",
            "itchiness of eye",
            "blindness",
            "eye burns or stings",
            "itchy eyelid",
            "feeling cold",
            "decreased appetite",
            "excessive appetite",
            "excessive anger",
            "loss of sensation",
            "focal weakness",
            "slurring words",
            "symptoms of the face",
            "disturbance of memory",
            "paresthesia",
            "side pain",
            "fever",
            "shoulder pain",
            "shoulder stiffness or tightness",
            "shoulder weakness",
            "arm cramps or spasms",
            "shoulder swelling",
            "tongue lesions",
            "leg cramps or spasms",
            "abnormal appearing tongue",
            "ache all over",
            "lower body pain",
            "problems during pregnancy",
            "spotting or bleeding during pregnancy",
            "cramps and spasms",
            "upper abdominal pain",
            "stomach bloating",
            "changes in stool appearance",
            "unusual color or odor to urine",
            "kidney mass",
            "swollen abdomen",
            "symptoms of prostate",
            "leg stiffness or tightness",
            "difficulty breathing",
            "rib pain",
            "joint pain",
            "muscle stiffness or tightness",
            "pallor",
            "hand or finger lump or mass",
            "chills",
            "groin pain",
            "fatigue",
            "abdominal distention",
            "regurgitation.1",
            "symptoms of the kidneys",
            "melena",
            "flushing",
            "coughing up sputum",
            "seizures",
            "delusions or hallucinations",
            "shoulder cramps or spasms",
            "joint stiffness or tightness",
            "pain or soreness of breast",
            "excessive urination at night",
            "bleeding from eye",
            "rectal bleeding",
            "constipation",
            "temper problems",
            "coryza",
            "wrist weakness",
            "eye strain",
            "hemoptysis",
            "lymphedema",
            "skin on leg or foot looks infected",
            "allergic reaction",
            "congestion in chest",
            "muscle swelling",
            "pus in urine",
            "abnormal size or shape of ear",
            "low back weakness",
            "sleepiness",
            "apnea",
            "abnormal breathing sounds",
            "excessive growth",
            "elbow cramps or spasms",
            "feeling hot and cold",
            "blood clots during menstrual periods",
            "absence of menstruation",
            "pulling at ears",
            "gum pain",
            "redness in ear",
            "fluid retention",
            "flu-like syndrome",
            "sinus congestion",
            "painful sinuses",
            "fears and phobias",
            "recent pregnancy",
            "uterine contractions",
            "burning chest pain",
            "back cramps or spasms",
            "stiffness all over",
            "muscle cramps, contractures, or spasms",
            "low back cramps or spasms",
            "back mass or lump",
            "nosebleed",
            "long menstrual periods",
            "heavy menstrual flow",
            "unpredictable menstruation",
            "painful menstruation",
            "infertility",
            "frequent menstruation",
            "sweating",
            "mass on eyelid",
            "swollen eye",
            "eyelid swelling",
            "eyelid lesion or rash",
            "unwanted hair",
            "symptoms of bladder",
            "irregular appearing nails",
            "itching of skin",
            "hurts to breath",
            "nailbiting",
            "skin dryness, peeling, scaliness, or roughness",
            "skin on arm or hand looks infected",
            "skin irritation",
            "itchy scalp",
            "hip swelling",
            "incontinence of stool",
            "foot or toe cramps or spasms",
            "warts",
            "bumps on penis",
            "too little hair",
            "foot or toe lump or mass",
            "skin rash",
            "mass or swelling around the anus",
            "low back swelling",
            "ankle swelling",
            "hip lump or mass",
            "drainage in throat",
            "dry or flaky scalp",
            "premenstrual tension or irritability",
            "feeling hot",
            "feet turned in",
            "foot or toe stiffness or tightness",
            "pelvic pressure",
            "elbow swelling",
            "elbow stiffness or tightness",
            "early or late onset of menopause",
            "mass on ear",
            "bleeding from ear",
            "hand or finger weakness",
            "low self-esteem",
            "throat irritation",
            "itching of the anus",
            "swollen or red tonsils",
            "irregular belly button",
            "swollen tongue",
            "lip sore",
            "vulvar sore",
            "hip stiffness or tightness",
            "mouth pain",
            "arm weakness",
            "leg lump or mass",
            "disturbance of smell or taste",
            "discharge in stools",
            "penis pain",
            "loss of sex drive",
            "obsessions and compulsions",
            "antisocial behavior",
            "neck cramps or spasms",
            "pupils unequal",
            "poor circulation",
            "thirst",
            "sleepwalking",
            "skin oiliness",
            "sneezing",
            "bladder mass",
            "knee cramps or spasms",
            "premature ejaculation",
            "leg weakness",
            "posture problems",
            "bleeding in mouth",
            "tongue bleeding",
            "change in skin mole size or color",
            "penis redness",
            "penile discharge",
            "shoulder lump or mass",
            "polyuria",
            "cloudy eye",
            "hysterical behavior",
            "arm lump or mass",
            "nightmares",
            "bleeding gums",
            "pain in gums",
            "bedwetting",
            "diaper rash",
            "lump or mass of breast",
            "vaginal bleeding after menopause",
            "infrequent menstruation",
            "mass on vulva",
            "jaw pain",
            "itching of scrotum",
            "postpartum problems of the breast",
            "eyelid retracted",
            "hesitancy",
            "elbow lump or mass",
            "muscle weakness",
            "throat redness",
            "joint swelling",
            "tongue pain",
            "redness in or around nose",
            "wrinkles on skin",
            "foot or toe weakness",
            "hand or finger cramps or spasms",
            "back stiffness or tightness",
            "wrist lump or mass",
            "skin pain",
            "low back stiffness or tightness",
            "low urine output",
            "skin on head or neck looks infected",
            "stuttering or stammering",
            "problems with orgasm",
            "nose deformity",
            "lump over jaw",
            "sore in nose",
            "hip weakness",
            "back swelling",
            "ankle stiffness or tightness",
            "ankle weakness",
            "neck weakness",
        ]

    return model, features


# Load model and features
model, all_symptoms = load_model_and_features()

# Main content split into two columns
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### Select Your Symptoms")

    # Use multiselect with all symptoms
    selected_symptoms = st.multiselect(
        "Select all that apply:", options=all_symptoms, default=[]
    )

    # Show selected symptoms count
    if selected_symptoms:
        st.info(f"You have selected {len(selected_symptoms)} symptoms")

with col2:
    st.markdown("### Prediction Results")

    if selected_symptoms:
        if not selected_symptoms:
            st.warning("Please select at least one symptom before predicting.")
        else:
            # Create input data frame
            input_data = pd.DataFrame(0, index=[0], columns=all_symptoms)

            # Set selected symptoms to 1
            for symptom in selected_symptoms:
                if symptom in input_data.columns:
                    input_data[symptom] = 1

            # Make prediction
            with st.spinner("Analyzing symptoms..."):
                try:
                    # Get prediction
                    predicted_disease = model.predict(input_data)[0]

                    # Get probabilities if available
                    if hasattr(model, "predict_proba"):
                        probabilities = model.predict_proba(input_data)[0]
                        top_indices = np.argsort(probabilities)[-3:][
                            ::-1
                        ]  # Get top 3 indices
                        top_diseases = [model.classes_[i] for i in top_indices]
                        top_probs = [probabilities[i] for i in top_indices]

                        # Display primary prediction
                        st.markdown(
                            f"<h2>Likely Disease: {predicted_disease}</h2>",
                            unsafe_allow_html=True,
                        )

                        # Display top 3 predictions with confidence
                        st.markdown("### Other possibilities:")
                        for disease, prob in zip(top_diseases, top_probs):
                            if prob > 0.05:  # Only show diseases with >5% probability
                                st.markdown(
                                    f"- **{disease}** (Confidence: {prob*100:.1f}%)"
                                )
                    else:
                        # If model doesn't have predict_proba, just show the prediction
                        st.markdown(
                            f"<h2>Predicted Disease: {predicted_disease}</h2>",
                            unsafe_allow_html=True,
                        )

                except Exception as e:
                    st.error(f"An error occurred: {e}")

            st.markdown("</div>", unsafe_allow_html=True)

            if predicted_disease:
                st.markdown("<div style='margin-top: 20px;'>", unsafe_allow_html=True)
                if st.button("Find Recommended Doctors", type="primary"):
                    st.session_state["disease"] = predicted_disease
                st.markdown("</div>", unsafe_allow_html=True)

            # Display selected symptoms
            st.markdown("### Your Selected Symptoms:")
            for i, symptom in enumerate(selected_symptoms, 1):
                st.markdown(f"{i}. {symptom}")

# Disclaimer
st.markdown(
    "<div class='disclaimer'>Disclaimer: This tool is for educational purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.</div>",
    unsafe_allow_html=True,
)

# Sidebar with information
with st.sidebar:
    st.markdown("<h1 style='text-align: center'>🏥</h1>", unsafe_allow_html=True)
    st.title("About This App")
    st.info(
        "This application uses machine learning to predict potential diseases "
        "based on symptoms. The predictions are based on a trained model and "
        "should be used for reference only."
    )

    st.markdown("### How to use:")
    st.markdown(
        """
        1. Type in the search box to filter symptoms
        2. Select all symptoms that apply to you
        3. Click the 'Predict Disease' button
        4. Review the predicted disease and confidence level
        """
    )

    st.markdown("### Model Information:")
    st.markdown(
        """
        - **Model Type**: Multinomial Naive Bayes
        - **Features**: " + str(len(all_symptoms)) + " symptoms
        """
    )
