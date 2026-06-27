import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Bike Sharing Demand Prediction App",
    layout="wide"
)

DATA_PATH = "bike_sharing_enhanced.csv"
OUTPUT_DIR = Path("model_outputs")

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

df = load_data()

st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Select a page",
    [
        "Business Case + Dataset",
        "Data Visualizations",
        "Prediction Models",
        "Feature Importance",
        "Model Comparison / Tuning",
        "Conclusion"
    ]
)

st.title("Bike Sharing Demand Prediction App")

if page == "Business Case + Dataset":
    st.header("Business Case")

    st.write("""
    The goal of this project is to predict hourly bike rental demand using
    weather, time, and calendar-related factors.
    """)

    st.write("""
    This problem matters because bike-sharing companies and cities need to know
    when demand will be high or low. Better demand prediction can help place the
    right number of bikes at the right time, improving bike availability for users.
    """)

    st.header("Social and Environmental Impact")

    st.write("""
    Bike sharing supports cleaner urban transportation by reducing car dependency,
    traffic congestion, and transportation-related emissions. Predicting demand can
    make bike sharing more reliable and encourage more people to use low-carbon
    transportation.
    """)

    st.header("Dataset Presentation")

    st.write("Dataset used: Bike Sharing Dataset")
    st.write("File used: hour.csv")
    st.write("Target variable: cnt")
    st.write(f"Rows: {df.shape[0]}")
    st.write(f"Columns: {df.shape[1]}")

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Columns Excluded From Modeling")

    st.write("""
    We excluded `instant`, `dteday`, `casual`, and `registered` from the model.
    `instant` is only an ID, `dteday` is used only for visualizations, and
    `casual + registered = cnt`, which would create data leakage.
    """)

elif page == "Data Visualizations":
    st.header("Data Visualizations")

    st.subheader("Average Bike Rentals by Hour")
    hourly = df.groupby("hr")["cnt"].mean()
    st.line_chart(hourly)

    st.write("""
    Bike demand is highest around commuting hours, especially in the morning and
    late afternoon.
    """)

    st.subheader("Average Bike Rentals by Month")
    monthly = df.groupby("Month")["cnt"].mean()
    st.bar_chart(monthly)

    st.subheader("Average Bike Rentals by Weekday")
    weekday = df.groupby("Weekday_Name")["cnt"].mean()
    st.bar_chart(weekday)

    st.subheader("Average Bike Rentals by Season")
    season = df.groupby("Season_Name")["cnt"].mean()
    st.bar_chart(season)

    st.subheader("Average Bike Rentals by Weather Condition")
    weather = df.groupby("Weather_Condition")["cnt"].mean()
    st.bar_chart(weather)

elif page == "Prediction Models":
    st.header("Prediction Models")

    st.write("""
    We trained two main models: Linear Regression and Random Forest Regressor.
    The target variable is `cnt`, which represents total hourly bike rentals.
    """)

    comparison_path = OUTPUT_DIR / "model_comparison.csv"

    if comparison_path.exists():
        comparison = pd.read_csv(comparison_path)
        st.subheader("Model Comparison")
        st.dataframe(comparison)
    else:
        st.warning("model_comparison.csv was not found. Run ML_Models.py first.")

    st.subheader("Final Result")

    st.write("""
    Random Forest performed better than Linear Regression because it had a lower
    MAE and RMSE and a higher R² score.
    """)

elif page == "Feature Importance":
    st.header("Feature Importance / Driving Variables")

    image_path = OUTPUT_DIR / "random_forest_feature_importance.png"

    if image_path.exists():
        st.image(str(image_path), caption="Top 15 Feature Importance - Random Forest")
    else:
        st.warning("Feature importance image was not found. Run ML_Models.py first.")

    st.write("""
    The most important variables were apparent temperature, hour of the day,
    humidity, and working day status. This means bike demand is mainly influenced
    by weather comfort and commuting patterns.
    """)

elif page == "Model Comparison / Tuning":
    st.header("Model Comparison and Hyperparameter Tuning")

    final_path = OUTPUT_DIR / "final_model_comparison.csv"

    if final_path.exists():
        final_comparison = pd.read_csv(final_path)
        st.dataframe(final_comparison)
    else:
        st.warning("final_model_comparison.csv was not found. Run ML_Models.py first.")

    st.write("""
    Hyperparameter tuning was tested using GridSearchCV. However, the tuned Random
    Forest did not improve performance compared to the original Random Forest.
    Therefore, the original Random Forest was selected as the final best model.
    """)

elif page == "Conclusion":
    st.header("Conclusion")

    st.write("""
    The Random Forest Regressor was the best-performing model. It achieved the
    lowest prediction error and the highest R² score, meaning it explained most of
    the variation in hourly bike rental demand.
    """)

    st.write("""
    The most important factors were apparent temperature, hour of the day, humidity,
    and working day status. These results make sense because people are more likely
    to rent bikes when the weather is comfortable and during commuting hours.
    """)

    st.write("""
    This app can help bike-sharing companies and cities better anticipate demand,
    improve bike availability, reduce car dependency, and support cleaner urban
    transportation.
    """)