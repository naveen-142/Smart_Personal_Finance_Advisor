import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==========================
# 🔹 Load Model & Features
# ==========================
ridge = joblib.load("Pickle/ridge_model.pkl")
model_features = joblib.load("Pickle/model_features.pkl")

# ==========================
# 🔹 Expense Columns
# ==========================
expense_cols = [
    "Rent", "Loan_Repayment", "Insurance", "Groceries", "Transport",
    "Eating_Out", "Entertainment", "Utilities", "Healthcare", "Education", "Miscellaneous"
]

# ==========================
# 🔹 Recommendation Function
# ==========================
def generate_recommendations(user):
    recs = []

    income = user["Income"] + user["Additional_Income"]
    total_exp = sum([user[col] for col in expense_cols])
    actual_savings = income - total_exp
    savings_rate = actual_savings / income
    debt_ratio = user["Loan_Repayment"] / income
    health_score = 0.4 * savings_rate + 0.3 * (1 - debt_ratio) + 0.3 * user["Credit_Score"] / 900

    flexible_expenses = ["Eating_Out", "Entertainment", "Groceries", "Miscellaneous", "Transport", "Utilities"]
    top_expense = max(flexible_expenses, key=lambda x: user[x])
    top_exp_value = user[top_expense]

    suggestion = f"Your highest flexible expense is {top_expense} at ₹{top_exp_value:.0f}. "
    if top_expense in ["Eating_Out", "Entertainment"]:
        suggestion += "Reduce by 20–30% through meal planning or fewer outings — the saved amount can go toward clearing your loan repayment faster."
    elif top_expense == "Groceries":
        suggestion += "Track grocery waste and buy in bulk; use savings to increase insurance or emergency fund."
    elif top_expense == "Transport":
        suggestion += "Try carpooling or public transport twice a week; redirect saved money into investments."
    elif top_expense == "Utilities":
        suggestion += "Switch to energy-efficient devices; the reduction can offset monthly insurance or EMI costs."
    elif top_expense == "Miscellaneous":
        suggestion += "Track impulsive buys. Saved cash can strengthen your emergency fund."
    else:
        suggestion += "Review this category for optimization."

    if savings_rate < 0:
        recs.append(f"⚠️ Your expenses exceed your income by ₹{-actual_savings:.0f}. Immediate cost reduction is required.")
    elif savings_rate < 0.2:
        recs.append(f"💡 You’re saving only {savings_rate*100:.1f}% of your income. Automate savings and reduce discretionary costs.")
    elif savings_rate < 0.35:
        recs.append(f"✅ Good savings rate ({savings_rate*100:.1f}%). Aim for 30–35% for better security.")
    else:
        recs.append(f"🌟 Excellent! You’re saving {savings_rate*100:.1f}%. Consider investing extra funds wisely.")

    if debt_ratio > 0.5:
        recs.append("📉 High debt load (>50% of income). Prioritize repaying high-interest loans first or consider refinancing options.")
    elif debt_ratio > 0.3:
        recs.append("⚠️ Moderate debt ratio. Focus on reducing EMIs before taking on new financial commitments.")
    else:
        recs.append("🟢 Low debt level — maintain this by avoiding unnecessary loans.")

    if user["Credit_Score"] < 600:
        recs.append("🔴 Low credit score — pay bills on time and limit card usage under 30%.")
    elif user["Credit_Score"] < 750:
        recs.append("🟠 Average credit score — avoid missed payments and monitor your score monthly.")
    else:
        recs.append("🟢 Excellent credit score — you qualify for low-interest offers; use wisely.")

    recs.append(suggestion)

    if health_score > 0.75:
        recs.append("💪 Financial health is strong! Explore mutual funds or SIPs for long-term wealth building.")
    elif health_score > 0.5:
        recs.append("📈 Financial health is okay. Reduce small expenses to improve cash flow stability.")
    else:
        recs.append("🚨 Financial health is weak. Cut discretionary costs and prioritize essential payments first.")

    return recs, health_score, top_expense

# ==========================
# 🔹 Streamlit UI (Improved Layout)
# ==========================
st.set_page_config(page_title="Smart Expense Advisor", page_icon="💰", layout="wide")

st.title("💰 Smart Expense Advisor")
st.caption("Analyze your monthly budget, predict savings, and get personalized cost-saving tips.")

st.markdown("---")

# =======================
# 👤 Personal & Income Info
# =======================
st.header("👤 Personal & Income Details")
col1, col2, col3 = st.columns(3)

with col1:
    income = st.number_input("💵 Monthly Income (₹)", min_value=0, step=1000)
    add_income = st.number_input("➕ Additional Income (₹)", min_value=0, step=500)
with col2:
    age = st.number_input("🎂 Age", min_value=18, max_value=100, step=1)
    dependents = st.number_input("👨‍👩‍👧 Dependents", min_value=0, step=1)
with col3:
    credit = st.number_input("💳 Credit Score", min_value=300, max_value=900, step=10)
    occupation = st.selectbox("🧑‍💼 Occupation", 
                              ["Salaried Employee", "Freelancer", "BusinessOwner", "Unemployed", "Retired"])

st.markdown("---")

# =======================
# 💸 Monthly Expenses
# =======================
st.header("💸 Monthly Expenses Breakdown")

with st.expander("🏠 Essential Expenses", expanded=True):
    c1, c2, c3 = st.columns(3)
    with c1:
        rent = st.number_input("🏘️ Rent", min_value=0, step=500)
        loan = st.number_input("💰 Loan Repayment", min_value=0, step=500)
    with c2:
        insurance = st.number_input("🩺 Insurance", min_value=0, step=500)
        groceries = st.number_input("🛒 Groceries", min_value=0, step=500)
    with c3:
        transport = st.number_input("🚗 Transport", min_value=0, step=500)
        utilities = st.number_input("⚡ Utilities", min_value=0, step=500)

with st.expander("🎉 Lifestyle & Discretionary Expenses", expanded=False):
    c4, c5, c6 = st.columns(3)
    with c4:
        eating_out = st.number_input("🍽️ Eating Out", min_value=0, step=500)
        entertainment = st.number_input("🎬 Entertainment", min_value=0, step=500)
    with c5:
        healthcare = st.number_input("🏥 Healthcare", min_value=0, step=500)
        education = st.number_input("📚 Education", min_value=0, step=500)
    with c6:
        misc = st.number_input("🧾 Miscellaneous", min_value=0, step=500)

st.markdown("---")

# =======================
# 🔍 Predict Button
# =======================
if st.button("🔍 Predict & Recommend", use_container_width=True):
    user_input = {
        "Income": income,
        "Age": age,
        "Dependents": dependents,
        "Rent": rent,
        "Loan_Repayment": loan,
        "Insurance": insurance,
        "Groceries": groceries,
        "Transport": transport,
        "Eating_Out": eating_out,
        "Entertainment": entertainment,
        "Utilities": utilities,
        "Healthcare": healthcare,
        "Education": education,
        "Miscellaneous": misc,
        "Additional_Income": add_income,
        "Credit_Score": credit,
        "Occupation": occupation
    }

    df = pd.DataFrame([user_input])
    df_X = pd.get_dummies(df, columns=["Occupation"], drop_first=True)
    for col in model_features:
        if col not in df_X.columns:
            df_X[col] = 0
    df_X = df_X[model_features]

    predicted_savings = ridge.predict(df_X)[0]
    recs, health_score, top_exp = generate_recommendations(user_input)

    # =======================
    # 📊 Results Display
    # =======================
    st.subheader("📊 Results Summary")
    colA, colB, colC = st.columns(3)
    with colA:
        st.success(f"💰 **Predicted Savings:** ₹{predicted_savings:,.2f}")
    with colB:
        st.info(f"⚡ **Top Expense to Cut:** {top_exp}")
    with colC:
        st.metric("🩺 Financial Health Score", f"{health_score*100:.1f}%")

    st.markdown("---")
    st.subheader("💡 Personalized Recommendations")
    for i, r in enumerate(recs, 1):
        st.write(f"**{i}.** {r}")

    # =======================
    # 📈 Visual Insights (Optional)
    # =======================
    st.markdown("---")
    st.subheader("📊 Expense Distribution Overview")

    expense_data = {col: user_input[col] for col in expense_cols}
    expense_df = pd.DataFrame(list(expense_data.items()), columns=["Category", "Amount"])
    expense_df = expense_df[expense_df["Amount"] > 0]

    if not expense_df.empty:
        import plotly.express as px
        fig = px.pie(expense_df, names="Category", values="Amount", 
                     title="Expense Breakdown by Category",
                     color_discrete_sequence=px.colors.qualitative.Set3)
        st.plotly_chart(fig, use_container_width=True)

        bar = px.bar(expense_df, x="Category", y="Amount", 
                     title="Category-wise Spending", 
                     color="Amount", text_auto=".2s")
        st.plotly_chart(bar, use_container_width=True)
