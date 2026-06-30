import streamlit as st
import pandas as pd
from pathlib import Path
import plotly.express as px

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

    st.header("Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Rows", f"{df.shape[0]:,}")

    with col2:
        st.metric("Columns", df.shape[1])

    with col3:
        st.metric("Target", "cnt")

    with col4:
        st.metric("Dataset", "hour.csv")

    st.subheader("Dataset Preview")

    st.write("The table below shows the first five observations of the dataset after preprocessing.")

    preview_cols = [
        "season",
        "yr",
        "mnth",
        "hr",
        "weekday",
        "workingday",
        "weathersit",
        "temp",
        "hum",
        "cnt"
    ]

    preview = df[preview_cols].head().copy()
    preview["temp"] = preview["temp"].round(2)
    preview["hum"] = preview["hum"].round(2)

    styled = (
        preview.style
        .hide(axis="index")
        .set_properties(**{"text-align": "center"})
        .set_table_styles([{"selector": "th", "props": [("text-align", "center")]}])
    )

    preview = (
         df[preview_cols]
        .head()
        .style
        .hide(axis="index")
        .set_properties(**{"text-align": "left", "border": "1px solid #cccccc"})
        .set_table_styles([
            {"selector": "th", "props": [("text-align", "left"), ("border", "1px solid #cccccc")]},
            {"selector": "table", "props": [("border-collapse", "collapse")]},
        ])
    )
    st.table(preview)
    
    st.markdown(
        "**Note:** The following variables were intentionally excluded to avoid data leakage or because they do not provide predictive value."
    )

    st.subheader("Columns Excluded From Modeling")

    st.markdown("""
- **instant** → Unique row identifier (not useful for prediction).
- **dteday** → Used only for visualization and date reference.
- **casual** → Excluded because it directly contributes to the target (`cnt`).
- **registered** → Excluded because it directly contributes to the target (`cnt`).
""")

elif page == "Data Visualizations":
    st.header("Data Visualizations")

    st.write("""
    This page explores how bike rental demand changes depending on time,
    month, weekday, season, and weather conditions.
    """)

    st.subheader("Average Bike Rentals by Hour")

    hourly = df.groupby("hr", as_index=False)["cnt"].mean()

    fig_hour = px.line(
        hourly,
        x="hr",
        y="cnt",
        markers=True,
        labels={"hr": "Hour of the Day", "cnt": "Average Bike Rentals"}
    )

    fig_hour.update_layout(xaxis=dict(tickmode="linear", dtick=1))
    st.plotly_chart(fig_hour, use_container_width=True)

    st.markdown("""
    **Insight:** Bike demand is highest during commuting hours, especially around
    **8 AM** and **5–6 PM**. This suggests that many users rely on bike sharing
    for work or school transportation.
    """)

    st.subheader("Average Bike Rentals by Month")

    month_order = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ]

    monthly = df.groupby("Month", as_index=False)["cnt"].mean()
    monthly["Month"] = pd.Categorical(monthly["Month"], categories=month_order, ordered=True)
    monthly = monthly.sort_values("Month")

    fig_month = px.bar(
        monthly,
        x="Month",
        y="cnt",
        text_auto=".0f",
        labels={"Month": "Month", "cnt": "Average Bike Rentals"}
    )

    st.plotly_chart(fig_month, use_container_width=True)

    st.markdown("""
**Insight:** Bike rentals increase from spring into summer, with the highest demand around June through September. Demand drops during winter months, especially in January and December.
""")

    st.subheader("Average Bike Rentals by Weekday")

    weekday_order = [
        "Sunday", "Monday", "Tuesday", "Wednesday",
        "Thursday", "Friday", "Saturday"
    ]

    weekday = df.groupby("Weekday_Name", as_index=False)["cnt"].mean()
    weekday["Weekday_Name"] = pd.Categorical(
        weekday["Weekday_Name"],
        categories=weekday_order,
        ordered=True
    )
    weekday = weekday.sort_values("Weekday_Name")

    fig_weekday = px.bar(
        weekday,
        x="Weekday_Name",
        y="cnt",
        text_auto=".0f",
        labels={"Weekday_Name": "Weekday", "cnt": "Average Bike Rentals"}
    )

    st.plotly_chart(fig_weekday, use_container_width=True)

    st.markdown("""
**Insight:** Bike rentals are fairly consistent across weekdays. This suggests that weekday alone is not the strongest predictor, but it may still help capture small differences in commuting behavior.
""")

    st.subheader("Average Bike Rentals by Season")

    season_order = ["Spring", "Summer", "Fall", "Winter"]


    season_fix = {
        "Spring": "Winter",
        "Summer": "Spring",
        "Fall": "Summer",
        "Winter": "Fall"
    }
    season = df.copy()
    season["Season_Name"] = season["Season_Name"].map(season_fix)
    season = season.groupby("Season_Name", as_index=False)["cnt"].mean()
    
    season["Season_Name"] = pd.Categorical(
        season["Season_Name"],
        categories=season_order,
        ordered=True
    )
    season = season.sort_values("Season_Name")

    fig_season = px.bar(
        season,
        x="Season_Name",
        y="cnt",
        text_auto=".0f",
        labels={"Season_Name": "Season", "cnt": "Average Bike Rentals"}
    )

    st.plotly_chart(fig_season, use_container_width=True)

    st.markdown("""
    **Insight:** Demand is lowest in winter and peaks in summer. Warmer, more comfortable conditions encourage more people to ride.
    """)

    st.subheader("Average Bike Rentals by Weather Condition")

    weather_order = [
        "Clear or Partly Cloudy",
        "Mist or Cloudy",
        "Light Snow or Light Rain",
        "Heavy Rain or Snow"
    ]

    weather = df.groupby("Weather_Condition", as_index=False)["cnt"].mean()
    weather["Weather_Condition"] = pd.Categorical(
        weather["Weather_Condition"],
        categories=weather_order,
        ordered=True
    )
    weather = weather.sort_values("Weather_Condition")

    fig_weather = px.bar(
        weather,
        x="Weather_Condition",
        y="cnt",
        text_auto=".0f",
        labels={
            "Weather_Condition": "Weather Condition",
            "cnt": "Average Bike Rentals"
        }
    )

    st.plotly_chart(fig_weather, use_container_width=True)

    st.markdown("""
    **Insight:** Clear or partly cloudy weather has the highest rental demand.
    Poor weather conditions reduce demand because people are less likely to ride bikes
    during rain, snow, or uncomfortable weather.
    """)

elif page == "Prediction Models":
    st.header("Prediction Models")

    st.write("""
    We trained two main models: Linear Regression and Random Forest Regressor.
    The target variable is **cnt**, which represents total hourly bike rentals.
    """)

    comparison_path = OUTPUT_DIR / "model_comparison.csv"

    if comparison_path.exists():
        comparison = pd.read_csv(comparison_path)

        comparison = comparison.round(3)
        comparison = comparison.reset_index(drop=True)

        styled_comparison = (
            comparison.style
            .hide(axis="index")
            .format(precision=3)
            .set_properties(**{"text-align": "left", "border": "1px solid #cccccc"})
            .set_table_styles([
                {"selector": "th", "props": [("text-align", "left"), ("border", "1px solid #cccccc")]},
                {"selector": "td", "props": [("text-align", "left")]},
                {"selector": "table", "props": [("border-collapse", "collapse"), ("width", "auto")]},
            ])
        )
        st.subheader("Model Comparison")
        st.table(styled_comparison)

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
        st.image(str(image_path))
    else:
        st.warning("Feature importance image was not found. Run ML_Models.py first.")

    st.write("""
    The most important variables were apparent temperature, specific commuting
    hours (such as 5 PM and 8 AM), and humidity. This means bike demand is mainly
    influenced by weather comfort and commuting patterns.
    """)

elif page == "Model Comparison / Tuning":
    st.header("Model Comparison and Hyperparameter Tuning")

    final_path = OUTPUT_DIR / "final_model_comparison.csv"

    if final_path.exists():
        final_comparison = pd.read_csv(final_path)
        styled_final = (
            final_comparison.style
            .hide(axis="index")
            .format(precision=3)
            .set_properties(**{"text-align": "left", "border": "1px solid #cccccc"})
            .set_table_styles([
                {"selector": "th", "props": [("text-align", "left"), ("border", "1px solid #cccccc")]},
                {"selector": "td", "props": [("text-align", "left")]},
                {"selector": "table", "props": [("border-collapse", "collapse"), ("width", "auto")]},
            ])
        )
        st.table(styled_final)
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
