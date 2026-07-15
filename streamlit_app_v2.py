"""
Bank Customer Churn Risk Dashboard
===================================
Run with:  streamlit run streamlit_app_v2.py
Requires:  streamlit pandas numpy matplotlib seaborn shap randomforest scikit-learn joblib plotly
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import shap, joblib, warnings, os
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title  = "Churn Risk Dashboard",
    page_icon   = "🏦",
    layout      = "wide",
    initial_sidebar_state = "expanded"
)

# ── Load artefacts ────────────────────────────────────────────
@st.cache_resource
def load_artefacts():
    data  = joblib.load("data/processed/churn_preprocessed.pkl")
    model = joblib.load("models/best_model.pkl")
    meta  = joblib.load("models/model_metadata.pkl")
    expl  = shap.TreeExplainer(model)
    return data, model, meta, expl


def extract_shap_class1(shap_values, explainer):
    """
    Extract class-1 (Churn) SHAP values and base value.
    Handles both output shapes from TreeExplainer:
      - list [class0_arr, class1_arr]          — older SHAP / RF
      - 3D ndarray (N, features, 2)            — newer SHAP / RF
      - 2D ndarray (N, features)               — XGBoost binary
    """
    sv_arr = np.array(shap_values)
    if isinstance(shap_values, list):
        # Older SHAP: list of two arrays
        sv       = shap_values[1]
        base_val = float(np.array(explainer.expected_value).flat[1])
    elif sv_arr.ndim == 3:
        # Newer SHAP: single 3D array (N, features, classes)
        sv       = sv_arr[:, :, 1]
        base_val = float(np.array(explainer.expected_value).flat[1])
    else:
        # Binary output already 2D — XGBoost default
        sv       = sv_arr
        base_val = float(np.array(explainer.expected_value).flat[-1])
    return sv, base_val

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/7/7e/ECB_star_logo.svg/200px-ECB_star_logo.svg.png",
             width=80)
    st.title("🏦 Churn Intelligence")
    st.caption("European Central Bank — Retail Analytics")
    st.divider()
    page = st.radio(
        "Navigate",
        ["📊 Overview Dashboard",
         "🔍 Customer Risk Calculator",
         "🎯 Feature Importance",
         "🔬 What-If Simulator"],
        index=0
    )

# ── Helper: risk badge ────────────────────────────────────────
def risk_badge(prob):
    if prob >= 0.70:
        return "🔴 High Risk", "error"
    elif prob >= 0.40:
        return "🟠 Medium Risk", "warning"
    else:
        return "🟢 Low Risk", "success"

# ── Feature engineering (must match notebook) ─────────────────
def engineer_features(df):
    df = df.copy()
    df["BalanceToSalary"]      = df["Balance"] / (df["EstimatedSalary"] + 1)
    df["ProductDensity"]       = df["NumOfProducts"] / (df["Tenure"] + 1)
    df["EngagementProduct"]    = df["IsActiveMember"] * df["NumOfProducts"]
    df["AgeTenureInteraction"] = df["Age"] * df["Tenure"]
    df["IsZeroBalance"]        = (df["Balance"] == 0).astype(int)
    df["CreditScoreBucket"]    = pd.cut(df["CreditScore"],
                                         bins=[0,579,669,739,799,900],
                                         labels=["Poor","Fair","Good","VGood","Excellent"])
    df["CreditScoreBucket"]    = df["CreditScoreBucket"].astype(str)
    return df

def encode_customer(raw_dict, feature_cols):
    """Convert raw customer inputs to model-ready feature vector."""
    df = pd.DataFrame([raw_dict])
    df = engineer_features(df)
    df = pd.get_dummies(df, columns=["Geography","Gender","CreditScoreBucket"],
                         drop_first=False)
    # Align columns
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    return df[feature_cols]

# ─────────────────────────────────────────────────────────────
try:
    data, model, meta, explainer = load_artefacts()
    X_train      = data["X_train"]
    X_test       = data["X_test"]
    y_test       = data["y_test"]
    feature_cols = data["feature_cols"]
    best_model   = meta["name"]
    results_df   = meta["results"]

    y_prob_test = model.predict_proba(X_test)[:, 1]

except FileNotFoundError:
    st.error("⚠️ Artefacts not found. Run all notebooks first to generate `data/processed/` and `models/`.")
    st.stop()

# ═════════════════════════════════════════════════════════════
# PAGE 1 — Overview Dashboard
# ═════════════════════════════════════════════════════════════
if page == "📊 Overview Dashboard":

    st.title("📊 Churn Risk Overview")
    st.caption(f"Best model: **{best_model}**")

    # ── KPI row ───────────────────────────────────────────────
    total      = len(y_prob_test)
    high_risk  = int((y_prob_test >= 0.70).sum())
    med_risk   = int(((y_prob_test >= 0.40) & (y_prob_test < 0.70)).sum())
    low_risk   = int((y_prob_test < 0.40).sum())
    avg_prob   = float(y_prob_test.mean())

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Customers",   f"{total:,}")
    k2.metric("🔴 High Risk",      f"{high_risk:,}",  f"{high_risk/total*100:.1f}%")
    k3.metric("🟠 Medium Risk",    f"{med_risk:,}",   f"{med_risk/total*100:.1f}%")
    k4.metric("🟢 Low Risk",       f"{low_risk:,}",   f"{low_risk/total*100:.1f}%")
    k5.metric("Avg Churn Prob",    f"{avg_prob:.1%}")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        # Churn probability distribution
        fig = px.histogram(
            x=y_prob_test, nbins=50,
            color_discrete_sequence=["#4A90D9"],
            labels={"x": "Churn Probability"},
            title="Churn Probability Distribution"
        )
        fig.add_vline(x=0.40, line_dash="dash", line_color="orange",
                      annotation_text="Medium threshold (0.40)")
        fig.add_vline(x=0.70, line_dash="dash", line_color="red",
                      annotation_text="High threshold (0.70)")
        fig.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Risk tier donut
        tier_counts = {"High Risk": high_risk, "Medium Risk": med_risk, "Low Risk": low_risk}
        fig2 = go.Figure(go.Pie(
            labels=list(tier_counts.keys()),
            values=list(tier_counts.values()),
            hole=0.5,
            marker_colors=["#E74C3C","#F39C12","#2ECC71"],
        ))
        fig2.update_layout(title="Risk Tier Breakdown", height=350)
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    # Model comparison table
    st.subheader("Model Performance Comparison")
    st.dataframe(
        results_df.sort_values("AUC", ascending=False).style
            .background_gradient(subset=["AUC","F1","Recall"], cmap="YlOrRd")
            .format({"AUC":"{:.4f}","Accuracy":"{:.4f}","Precision":"{:.4f}",
                     "Recall":"{:.4f}","F1":"{:.4f}","CV_AUC":"{:.4f}"}),
        use_container_width=True, height=230
    )

# ═════════════════════════════════════════════════════════════
# PAGE 2 — Customer Risk Calculator
# ═════════════════════════════════════════════════════════════
elif page == "🔍 Customer Risk Calculator":

    st.title("🔍 Customer Churn Risk Calculator")
    st.caption("Enter customer details to get a real-time churn probability score.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Demographics")
        age         = st.slider("Age", 18, 92, 38)
        gender      = st.selectbox("Gender", ["Female", "Male"])
        geography   = st.selectbox("Geography", ["France", "Germany", "Spain"])
        credit_score = st.slider("Credit Score", 350, 850, 650)

    with col2:
        st.subheader("Financial Profile")
        balance          = st.number_input("Account Balance (€)", 0, 300000, 60000, step=1000)
        estimated_salary = st.number_input("Estimated Salary (€)", 10000, 200000, 60000, step=1000)
        num_products     = st.slider("Number of Products", 1, 4, 2)
        tenure           = st.slider("Tenure (years)", 0, 10, 5)

    with col3:
        st.subheader("Engagement")
        has_cr_card    = st.selectbox("Has Credit Card", [1, 0], format_func=lambda x: "Yes" if x else "No")
        is_active      = st.selectbox("Is Active Member", [1, 0], format_func=lambda x: "Yes" if x else "No")

    if st.button("🔮 Calculate Churn Risk", type="primary", use_container_width=True):
        raw = {
            "CreditScore": credit_score, "Geography": geography,
            "Gender": gender, "Age": age, "Tenure": tenure,
            "Balance": balance, "NumOfProducts": num_products,
            "HasCrCard": has_cr_card, "IsActiveMember": is_active,
            "EstimatedSalary": estimated_salary
        }
        X_customer = encode_customer(raw, feature_cols)
        churn_prob = float(model.predict_proba(X_customer)[0, 1])
        label, level = risk_badge(churn_prob)

        st.divider()
        r1, r2, r3 = st.columns(3)
        r1.metric("Churn Probability", f"{churn_prob:.1%}")
        r2.metric("Risk Score",        f"{churn_prob*100:.0f} / 100")
        r3.metric("Risk Tier",         label)

        # Gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode  = "gauge+number+delta",
            value = churn_prob * 100,
            title = {"text": "Churn Risk Score"},
            delta = {"reference": 20, "suffix": "% vs avg"},
            gauge = {
                "axis"     : {"range": [0, 100]},
                "bar"      : {"color": "#E74C3C" if churn_prob >= 0.7 else
                                        "#F39C12" if churn_prob >= 0.4 else "#2ECC71"},
                "steps"    : [{"range": [0, 40],  "color": "#D5F5E3"},
                               {"range": [40, 70], "color": "#FEF9E7"},
                               {"range": [70, 100],"color": "#FDEDEC"}],
                "threshold": {"line": {"color": "red", "width": 4}, "value": 70}
            }
        ))
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)

        # SHAP waterfall for this customer
        st.subheader("Why this prediction?")
        raw_sv   = explainer.shap_values(X_customer)
        sv_cust, base_val_cust = extract_shap_class1(raw_sv, explainer)
        # sv_cust shape: (1, features) — take row 0 for the single customer
        fig_shap, ax = plt.subplots(figsize=(10, 5))
        shap.waterfall_plot(
            shap.Explanation(
                values        = sv_cust[0],           # shape (features,) — class-1 slice
                base_values   = base_val_cust,
                data          = X_customer.iloc[0].values,
                feature_names = feature_cols
            ), show=False, max_display=12
        )
        plt.tight_layout()
        st.pyplot(fig_shap)
        plt.close()

# ═════════════════════════════════════════════════════════════
# PAGE 3 — Feature Importance Dashboard
# ═════════════════════════════════════════════════════════════
elif page == "🎯 Feature Importance":

    st.title("🎯 Feature Importance Dashboard")

    tab1, tab2, tab3 = st.tabs(["Built-in Importance", "SHAP Global", "SHAP Beeswarm"])

    with tab1:
        if hasattr(model, "feature_importances_"):
            imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
            top_n = st.slider("Show top N features", 5, len(feature_cols), 20)
            fig = px.bar(
                x=imp.head(top_n).values[::-1],
                y=imp.head(top_n).index[::-1],
                orientation="h",
                color=imp.head(top_n).values[::-1],
                color_continuous_scale="Blues",
                labels={"x":"Importance","y":"Feature"},
                title=f"{best_model} — Built-in Feature Importance (Top {top_n})"
            )
            fig.update_coloraxes(showscale=False)
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.info("Computing SHAP values on test set — may take a moment.")
        with st.spinner("Running SHAP..."):
            raw_sv_all       = explainer.shap_values(X_test)
            sv_all, base_val_all = extract_shap_class1(raw_sv_all, explainer)
            # sv_all shape: (N, features) — class-1 contributions only
            mean_abs = pd.Series(np.abs(sv_all).mean(axis=0), index=feature_cols).sort_values(ascending=False)

        top_n2 = st.slider("Show top N", 5, len(feature_cols), 20, key="shap_n")
        fig2 = px.bar(
            x=mean_abs.head(top_n2).values[::-1],
            y=mean_abs.head(top_n2).index[::-1],
            orientation="h",
            color=mean_abs.head(top_n2).values[::-1],
            color_continuous_scale="Oranges",
            labels={"x":"Mean |SHAP value|","y":"Feature"},
            title=f"SHAP Global Feature Importance (Top {top_n2})"
        )
        fig2.update_coloraxes(showscale=False)
        fig2.update_layout(height=600)
        st.plotly_chart(fig2, use_container_width=True)

        # Download ranking
        rank_df = mean_abs.reset_index()
        rank_df.columns = ["Feature","Mean_SHAP"]
        rank_df["Rank"] = range(1, len(rank_df)+1)
        st.download_button("⬇ Download SHAP ranking CSV",
                           rank_df.to_csv(index=False),
                           "shap_ranking.csv", "text/csv")

    with tab3:
        with st.spinner("Building beeswarm plot..."):
            # sv_all is already extracted in tab2; recompute if tab3 opened first
            if "sv_all" not in dir():
                raw_sv_all       = explainer.shap_values(X_test)
                sv_all, base_val_all = extract_shap_class1(raw_sv_all, explainer)
            fig_bee, ax = plt.subplots(figsize=(10, 8))
            shap.summary_plot(sv_all, X_test, feature_names=feature_cols,
                              show=False, max_display=20)
            plt.tight_layout()
            st.pyplot(fig_bee)
            plt.close()

# ═════════════════════════════════════════════════════════════
# PAGE 4 — What-If Simulator
# ═════════════════════════════════════════════════════════════
elif page == "🔬 What-If Simulator":

    st.title("🔬 What-If Scenario Simulator")
    st.caption("Adjust customer attributes and see how churn probability changes in real time.")

    st.info("Set a baseline customer, then modify one attribute at a time to see its impact.")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Baseline Customer")
        base_age      = st.slider("Age",               18, 92,     38, key="b_age")
        base_balance  = st.slider("Balance (€k)",       0, 300,     60, key="b_bal") * 1000
        base_products = st.slider("Num Products",        1,   4,     2, key="b_prod")
        base_active   = st.selectbox("Active Member",  [1, 0],       key="b_act",
                                     format_func=lambda x: "Yes" if x else "No")
        base_tenure   = st.slider("Tenure (years)",     0,  10,     5, key="b_ten")
        base_credit   = st.slider("Credit Score",     350, 850,   650, key="b_cred")
        base_geo      = st.selectbox("Geography", ["France","Germany","Spain"], key="b_geo")
        base_salary   = st.slider("Salary (€k)",       10, 200,    60, key="b_sal") * 1000

    def make_customer(age, balance, products, active, tenure, credit, geo, salary,
                      gender="Female", has_cr_card=1):
        raw = {
            "CreditScore": credit, "Geography": geo, "Gender": gender,
            "Age": age, "Tenure": tenure, "Balance": balance,
            "NumOfProducts": products, "HasCrCard": has_cr_card,
            "IsActiveMember": active, "EstimatedSalary": salary
        }
        X = encode_customer(raw, feature_cols)
        return float(model.predict_proba(X)[0, 1])

    base_prob = make_customer(base_age, base_balance, base_products, base_active,
                               base_tenure, base_credit, base_geo, base_salary)

    with col2:
        st.subheader("Live Churn Probability")
        lbl, lvl = risk_badge(base_prob)
        st.metric("Current Churn Probability", f"{base_prob:.1%}", label_visibility="visible")
        st.metric("Risk Tier", lbl)

        # Scenario sweep — Age
        st.divider()
        st.subheader("Sensitivity Analysis — Age")
        ages   = list(range(18, 92, 2))
        probs  = [make_customer(a, base_balance, base_products, base_active,
                                base_tenure, base_credit, base_geo, base_salary)
                  for a in ages]
        fig_age = px.line(x=ages, y=probs,
                          labels={"x":"Age","y":"Churn Probability"},
                          title="Churn Probability vs Age (all else equal)")
        fig_age.add_hline(y=0.40, line_dash="dash", line_color="orange")
        fig_age.add_hline(y=0.70, line_dash="dash", line_color="red")
        fig_age.add_vline(x=base_age, line_dash="dot", line_color="blue",
                          annotation_text=f"Current: {base_age}")
        fig_age.update_layout(height=280)
        st.plotly_chart(fig_age, use_container_width=True)

    # Full sensitivity matrix
    st.divider()
    st.subheader("Full Sensitivity — All Products & Activity Combinations")

    rows = []
    for prod in [1, 2, 3, 4]:
        for act in [0, 1]:
            prob = make_customer(base_age, base_balance, prod, act,
                                 base_tenure, base_credit, base_geo, base_salary)
            rows.append({"NumOfProducts": prod,
                         "IsActiveMember": "Active" if act else "Inactive",
                         "ChurnProbability": prob})

    sens_df = pd.DataFrame(rows)

    # ── Fix: use imshow on a pivot table so each cell shows ONE
    # probability value, not a sum. density_heatmap aggregates by
    # summing z-values per bin which produced invalid values like 106%.
    pivot = sens_df.pivot(
        index   = "IsActiveMember",
        columns = "NumOfProducts",
        values  = "ChurnProbability"
    )

    fig_heat = px.imshow(
        pivot,
        color_continuous_scale = "RdYlGn_r",
        zmin  = 0.0,
        zmax  = 1.0,
        title = "Churn Probability — Products × Activity",
        text_auto = ".1%",
        aspect = "auto",
        labels = {"x": "Num of Products", "y": "Membership Status",
                  "color": "Churn Probability"}
    )
    fig_heat.update_layout(height=300, coloraxis_colorbar=dict(tickformat=".0%"))
    st.plotly_chart(fig_heat, use_container_width=True)

    # Retention action suggestions
    st.divider()
    st.subheader("💡 Retention Action Suggestions")
    if base_prob >= 0.70:
        st.error("**High Risk** — Immediate intervention recommended.")
        st.markdown("""
- 📞 Assign a dedicated relationship manager
- 🎁 Offer personalised loyalty reward or fee waiver
- 💳 Cross-sell a complementary product (e.g. investment fund)
- 📱 Enable premium digital banking features
        """)
    elif base_prob >= 0.40:
        st.warning("**Medium Risk** — Proactive engagement advised.")
        st.markdown("""
- 📧 Send a personalised re-engagement email campaign
- 💰 Offer a competitive savings rate upgrade
- 📊 Provide a personalised financial review
        """)
    else:
        st.success("**Low Risk** — Customer is stable.")
        st.markdown("""
- 🏆 Enrol in loyalty programme
- 📈 Upsell premium products at next interaction
        """)

# ── Footer ─────────────────────────────────────────────────────
st.divider()
st.caption(f"🏦 Bank Churn Intelligence System · European Central Bank · Powered by {best_model} + SHAP")