import streamlit as st
import joblib


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Auto Ticket Classifier",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

    /* Main background */
    .stApp {
        background-color: #f5f7fb;
    }

    /* Main title */
    .main-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 17px;
        color: #6b7280;
        margin-bottom: 25px;
    }

    /* Header card */
    .header-card {
        background: white;
        padding: 28px;
        border-radius: 18px;
        border: 1px solid #e5e7eb;
        margin-bottom: 25px;
    }

    /* Result cards */
    .result-card {
        background: white;
        padding: 20px;
        border-radius: 14px;
        border: 1px solid #e5e7eb;
        min-height: 120px;
    }

    .result-label {
        font-size: 14px;
        color: #6b7280;
        margin-bottom: 8px;
    }

    .result-value {
        font-size: 22px;
        font-weight: 700;
    }

    /* Confidence card */
    .confidence-card {
        background: white;
        padding: 22px;
        border-radius: 16px;
        border: 1px solid #e5e7eb;
        margin-top: 20px;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #6b7280;
        font-size: 13px;
        padding: 30px 0 10px 0;
    }

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD MODELS
# =========================================================

@st.cache_resource
def load_models():

    category_model = joblib.load(
        "ticket_classifier.pkl"
    )

    subcategory_model = joblib.load(
        "subcategory_classifier.pkl"
    )

    priority_model = joblib.load(
        "priority_classifier.pkl"
    )

    assignment_model = joblib.load(
        "assignment_group_classifier.pkl"
    )

    tfidf = joblib.load(
        "tfidf_vectorizer.pkl"
    )

    return (
        category_model,
        subcategory_model,
        priority_model,
        assignment_model,
        tfidf
    )


category_model, subcategory_model, priority_model, assignment_model, tfidf = load_models()


# =========================================================
# HEADER
# =========================================================

st.markdown("""
<div class="header-card">

<div class="main-title">
🎫 Auto Ticket Classifier
</div>

<div class="subtitle">
AI-powered Employee IT Support Ticket Classification System
</div>

</div>
""", unsafe_allow_html=True)


# =========================================================
# TICKET INPUT
# =========================================================

st.subheader("📝 Create IT Support Ticket")

ticket = st.text_area(
    "Ticket Description",
    placeholder="Example: My VPN is not connecting from home",
    height=150,
    label_visibility="collapsed"
)


st.caption(
    "Enter a clear description of the employee's IT problem."
)


# =========================================================
# PREDICT BUTTON
# =========================================================

predict_button = st.button(
    "🔮  Predict Ticket",
    use_container_width=True
)


# =========================================================
# PREDICTION
# =========================================================

if predict_button:

    if not ticket.strip():

        st.warning(
            "⚠️ Please enter a ticket description first."
        )

    else:

        # -------------------------------------------------
        # TF-IDF
        # -------------------------------------------------

        ticket_tfidf = tfidf.transform(
            [ticket]
        )


        # -------------------------------------------------
        # CATEGORY
        # -------------------------------------------------

        category = category_model.predict(
            ticket_tfidf
        )[0]


        # -------------------------------------------------
        # CONFIDENCE
        # -------------------------------------------------

        probabilities = category_model.predict_proba(
            ticket_tfidf
        )[0]

        confidence = probabilities.max()


        # -------------------------------------------------
        # SUBCATEGORY
        # -------------------------------------------------

        subcategory = subcategory_model.predict(
            ticket_tfidf
        )[0]


        # -------------------------------------------------
        # PRIORITY
        # -------------------------------------------------

        priority = priority_model.predict(
            ticket_tfidf
        )[0]


        # -------------------------------------------------
        # ASSIGNMENT GROUP
        # -------------------------------------------------

        assignment = assignment_model.predict(
            ticket_tfidf
        )[0]


        # =================================================
        # RESULT SECTION
        # =================================================

        st.divider()

        st.subheader("🎯 Prediction Results")


        # -------------------------------------------------
        # RESULT CARDS
        # -------------------------------------------------

        col1, col2, col3, col4 = st.columns(4)


        with col1:

            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">
                    📂 Category
                    </div>
                    <div class="result-value">
                    {category}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


        with col2:

            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">
                    🗂️ Subcategory
                    </div>
                    <div class="result-value">
                    {subcategory}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


        with col3:

            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">
                    🚨 Priority
                    </div>
                    <div class="result-value">
                    {priority}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


        with col4:

            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">
                    👥 Assignment Group
                    </div>
                    <div class="result-value">
                    {assignment}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


        # =================================================
        # CONFIDENCE
        # =================================================

        st.markdown(
            '<div class="confidence-card">',
            unsafe_allow_html=True
        )

        st.subheader("🤖 Model Confidence")

        confidence_percentage = confidence * 100

        st.metric(
            "Category Confidence",
            f"{confidence_percentage:.2f}%"
        )

        st.progress(
            float(confidence)
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


        # =================================================
        # AUTO ASSIGN / HUMAN REVIEW
        # =================================================

        THRESHOLD = 0.80


        if confidence >= THRESHOLD:

            st.success(
                "✅ AUTO ASSIGN"
            )

            st.write(
                "The model has high confidence in this prediction. "
                "The ticket can be automatically assigned."
            )

        else:

            st.warning(
                "⚠️ HUMAN REVIEW REQUIRED"
            )

            st.write(
                "The model confidence is below the 80% threshold. "
                "A support agent should review this ticket."
            )


        # =================================================
        # TICKET SUMMARY
        # =================================================

        with st.expander("📄 View Ticket Details"):

            st.write(
                "**Original Ticket:**"
            )

            st.info(ticket)

            st.write(
                "**Predicted Category:**",
                category
            )

            st.write(
                "**Predicted Subcategory:**",
                subcategory
            )

            st.write(
                "**Priority:**",
                priority
            )

            st.write(
                "**Assignment Group:**",
                assignment
            )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">
        Auto Ticket Classifier • AI-powered IT Support Automation
    </div>
    """,
    unsafe_allow_html=True
)