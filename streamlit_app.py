import streamlit as st
import pandas as pd
from pathlib import Path
import plotly.express as px
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import ParameterGrid


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

    st.dataframe(df[preview_cols].head(), use_container_width=True)

    st.markdown(
        "**Note:** The following variables were intentionally excluded to avoid data leakage."
    )

    st.subheader("Columns Excluded From Modeling")

    st.markdown("""
- **instant** -> Unique row identifier (not useful for prediction).
- **dteday** -> Used only for visualization and date reference.
- **casual** -> Excluded because it directly contributes to the target (`cnt`).
- **registered** -> Excluded because it directly contributes to the target (`cnt`).
""")


elif page == "Data Visualizations":
    st.header("Data Visualizations")

    st.write("""
This page explores how bike rental demand changes depending on time,
month, weekday, season, and weather conditions. Use the filters below
to narrow down the data shown in every chart on this page.
""")

    st.subheader("Filters")

    filter_col1, filter_col2 = st.columns(2)

    season_filter_order = ["Spring", "Summer", "Fall", "Winter"]
    weather_filter_order = [
        "Clear or Partly Cloudy",
        "Mist or Cloudy",
        "Light Snow or Light Rain",
        "Heavy Rain or Snow"
    ]

    with filter_col1:
        selected_seasons = st.multiselect(
            "Season",
            options=season_filter_order,
            default=season_filter_order
        )

    with filter_col2:
        selected_weather = st.multiselect(
            "Weather Condition",
            options=weather_filter_order,
            default=weather_filter_order
        )

    df_viz = df.copy()

    if selected_seasons:
        df_viz = df_viz[df_viz["Season_Name"].isin(selected_seasons)]

    if selected_weather:
        df_viz = df_viz[df_viz["Weather_Condition"].isin(selected_weather)]

    st.caption(
        f"Showing {len(df_viz):,} of {len(df):,} hourly records based on current filters."
    )

    st.divider()

    st.subheader("Distribution of Hourly Bike Rentals")

    fig_dist = px.histogram(
        df_viz,
        x="cnt",
        nbins=50,
        labels={"cnt": "Hourly Bike Rentals"},
        color_discrete_sequence=["#3498db"]
    )
    fig_dist.update_layout(yaxis_title="Number of Hours", bargap=0.05)
    st.plotly_chart(fig_dist, use_container_width=True)

    st.markdown("""
**Insight:** Most hours see relatively low rental counts, with a long right tail
of high-demand hours. This skew is part of why Random Forest, which can capture
non-linear demand spikes, outperforms Linear Regression on this data.
""")

    st.subheader("Average Bike Rentals by Hour")

    hourly = df_viz.groupby("hr", as_index=False)["cnt"].mean()

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
**8 AM** and **5-6 PM**. This suggests that many users rely on bike sharing
for work or school transportation.
""")

    st.subheader("Average Bike Rentals by Month")

    month_order = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ]

    monthly = df_viz.groupby("Month", as_index=False)["cnt"].mean()
    monthly["Month"] = pd.Categorical(monthly["Month"], categories=month_order, ordered=True)
    monthly = monthly.sort_values("Month")

    fig_month = px.bar(
        monthly,
        x="Month",
        y="cnt",
        labels={"Month": "Month", "cnt": "Average Bike Rentals"}
    )

    st.plotly_chart(fig_month, use_container_width=True)

    st.markdown("""
**Insight:** Bike rentals increase from spring into summer, with the highest demand
around June through September. Warmer months make biking more comfortable and practical.
""")

    st.subheader("Average Bike Rentals by Weekday")

    weekday_order = [
        "Sunday", "Monday", "Tuesday", "Wednesday",
        "Thursday", "Friday", "Saturday"
    ]

    weekday = df_viz.groupby("Weekday_Name", as_index=False)["cnt"].mean()
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
        labels={"Weekday_Name": "Weekday", "cnt": "Average Bike Rentals"}
    )

    st.plotly_chart(fig_weekday, use_container_width=True)

    st.markdown("""
**Insight:** Bike rentals are fairly consistent across weekdays. This suggests that
both commuting and leisure trips contribute to demand throughout the week.
""")

    st.subheader("Average Bike Rentals by Season")

    season_order = ["Spring", "Summer", "Fall", "Winter"]

    season = df_viz.groupby("Season_Name", as_index=False)["cnt"].mean()
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
        labels={"Season_Name": "Season", "cnt": "Average Bike Rentals"}
    )

    st.plotly_chart(fig_season, use_container_width=True)

    st.markdown("""
**Insight:** Demand is lowest in spring and higher during warmer seasons.
Comfortable outdoor conditions increase bike usage.
""")

    st.subheader("Average Bike Rentals by Weather Condition")

    weather_order = [
        "Clear or Partly Cloudy",
        "Mist or Cloudy",
        "Light Snow or Light Rain",
        "Heavy Rain or Snow"
    ]

    weather = df_viz.groupby("Weather_Condition", as_index=False)["cnt"].mean()
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

    st.subheader("Average Bike Rentals by Day Type")

    def label_day(row):
        if row["holiday"] == 1:
            return "Holiday"
        elif row["workingday"] == 1:
            return "Working Day"
        else:
            return "Weekend / Non-working"

    df_daytype = df_viz.copy()
    df_daytype["Day_Type"] = df_daytype.apply(label_day, axis=1)

    daytype_order = ["Working Day", "Weekend / Non-working", "Holiday"]
    daytype_grouped = df_daytype.groupby("Day_Type", as_index=False)["cnt"].mean()
    daytype_grouped["Day_Type"] = pd.Categorical(
        daytype_grouped["Day_Type"],
        categories=daytype_order,
        ordered=True
    )
    daytype_grouped = daytype_grouped.sort_values("Day_Type")

    fig_daytype = px.bar(
        daytype_grouped,
        x="Day_Type",
        y="cnt",
        labels={"Day_Type": "Day Type", "cnt": "Average Bike Rentals"},
        color="Day_Type",
        color_discrete_sequence=["#3498db", "#2ecc71", "#e74c3c"]
    )
    fig_daytype.update_layout(showlegend=False)
    st.plotly_chart(fig_daytype, use_container_width=True)

    st.markdown("""
**Insight:** Working days show strong commuter-driven demand, while holidays
tend to follow a different pattern, often lower and more leisure-oriented.
This distinction matters for fleet planning, since bike redistribution needs
differ between commuter peaks and holiday leisure usage.
""")

    st.divider()

    st.subheader("Demand vs. Temperature, Humidity, and Wind Speed")

    weather_tab1, weather_tab2, weather_tab3 = st.tabs(
        ["Temperature", "Humidity", "Wind Speed"]
    )

    with weather_tab1:
        fig_temp_scatter = px.scatter(
            df_viz,
            x="atemp",
            y="cnt",
            color="Season_Name",
            opacity=0.35,
            labels={
                "atemp": "Feeling Temperature (normalized)",
                "cnt": "Bike Rentals",
                "Season_Name": "Season"
            },
            category_orders={"Season_Name": season_filter_order}
        )
        st.plotly_chart(fig_temp_scatter, use_container_width=True)
        st.markdown("""
**Insight:** Rentals generally increase as the feeling temperature rises, up to a
comfortable point, then level off. The relationship is not purely linear, which
is part of why Random Forest captures it better than Linear Regression.
""")

    with weather_tab2:
        fig_hum_scatter = px.scatter(
            df_viz,
            x="hum",
            y="cnt",
            color="Season_Name",
            opacity=0.35,
            labels={
                "hum": "Humidity (normalized)",
                "cnt": "Bike Rentals",
                "Season_Name": "Season"
            },
            category_orders={"Season_Name": season_filter_order}
        )
        st.plotly_chart(fig_hum_scatter, use_container_width=True)
        st.markdown("""
**Insight:** Very high humidity is associated with lower rental counts,
consistent with riders avoiding muggy or rainy conditions.
""")

    with weather_tab3:
        fig_wind_scatter = px.scatter(
            df_viz,
            x="windspeed",
            y="cnt",
            color="Season_Name",
            opacity=0.35,
            labels={
                "windspeed": "Wind Speed (normalized)",
                "cnt": "Bike Rentals",
                "Season_Name": "Season"
            },
            category_orders={"Season_Name": season_filter_order}
        )
        st.plotly_chart(fig_wind_scatter, use_container_width=True)
        st.markdown("""
**Insight:** Higher wind speeds show a mild negative relationship with demand,
though the effect is weaker than temperature or humidity.
""")

    st.divider()

    st.subheader("Correlation Heatmap")

    numeric_cols_for_corr = ["temp", "atemp", "hum", "windspeed", "hr", "mnth", "cnt"]
    available_corr_cols = [c for c in numeric_cols_for_corr if c in df_viz.columns]

    corr_matrix = df_viz[available_corr_cols].corr().round(2)

    fig_corr = px.imshow(
        corr_matrix,
        text_auto=True,
        color_continuous_scale="Blues",
        aspect="auto",
        labels={"color": "Correlation"}
    )
    fig_corr.update_layout(margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown("""
**Insight:** `temp` and `atemp` are very strongly correlated with each other,
which is expected since "feels like" temperature is derived from actual
temperature. `hr` shows a moderate relationship with `cnt`, reflecting the
clear commuting-hour pattern seen above. `hum` and `windspeed` show weaker,
negative correlations with demand.
""")


elif page == "Prediction Models":
    st.header("Prediction Models")

    st.write("""
We trained two models to predict hourly bike rental demand (cnt).
Use the selector below to explore each model and make your own live prediction.
""")

    model_choice = st.radio(
        "Select a model to explore:",
        ["Linear Regression", "Random Forest"],
        horizontal=True
    )

    if model_choice == "Linear Regression":
        model_file = OUTPUT_DIR / "linear_regression_model.pkl"
        plot_file = OUTPUT_DIR / "linear_regression_actual_vs_predicted.png"
        mae, rmse, r2 = 69.22, 97.75, 0.698
    else:
        model_file = OUTPUT_DIR / "random_forest_model.pkl"
        plot_file = OUTPUT_DIR / "random_forest_actual_vs_predicted.png"
        mae, rmse, r2 = 29.70, 47.59, 0.928

    st.subheader("Model Performance")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("MAE", f"{mae:.2f} rentals")

    with col2:
        st.metric("RMSE", f"{rmse:.2f} rentals")

    with col3:
        st.metric("R2 Score", f"{r2:.3f}")

    st.subheader("Actual vs Predicted Rentals")

    if plot_file.exists():
        st.image(str(plot_file), use_container_width=True)
    else:
        st.warning("Plot not found. Make sure the model_outputs folder is present.")

    if model_choice == "Linear Regression":
        st.markdown("""
**Reading this chart:** The red line represents a perfect prediction.
Points far from the line are prediction errors. Linear Regression struggles
with peak demand hours because bike demand spikes in a non-linear way.
""")
    else:
        st.markdown("""
**Reading this chart:** Random Forest predictions cluster much closer
to the red line, meaning it captures demand patterns
that Linear Regression cannot, like rush hour spikes and weather effects.
""")

    st.subheader("Residual Plot")

    model = joblib.load(model_file)

    df["is_weekend"] = df["weekday"].apply(lambda x: 1 if x in [0, 6] else 0)

    features = [
        "season", "yr", "mnth", "hr", "holiday", "weekday",
        "workingday", "weathersit", "is_weekend",
        "temp", "atemp", "hum", "windspeed"
    ]

    X = df[features]
    y = df["cnt"]

    preds = model.predict(X)
    preds = np.maximum(preds, 0)
    residuals = y - preds

    fig_resid = px.scatter(
        x=preds,
        y=residuals,
        labels={"x": "Predicted Rentals", "y": "Residuals (Actual - Predicted)"},
        opacity=0.3
    )

    fig_resid.add_hline(y=0, line_color="red")
    st.plotly_chart(fig_resid, use_container_width=True)

    if model_choice == "Linear Regression":
        st.markdown("""
**Reading this chart:** A good model has residuals scattered randomly around 0.
Linear Regression shows a clear fan-shaped pattern because errors get larger at
higher demand values, confirming it struggles with peak hours.
""")
    else:
        st.markdown("""
**Reading this chart:** Random Forest residuals are much more tightly clustered
around 0 across all predicted values, confirming it handles both low
and high demand hours well.
""")

    st.subheader("Try It Yourself! Live Prediction")
    st.write("Adjust the inputs below to describe a specific hour, then click Predict.")

    col1, col2, col3 = st.columns(3)

    with col1:
        hr = st.slider("Hour of Day", 0, 23, 8)
        temp = st.slider("Temperature (0-1)", 0.0, 1.0, 0.5, 0.01)
        atemp = st.slider("Feeling Temperature (0-1)", 0.0, 1.0, 0.5, 0.01)
        hum = st.slider("Humidity (0-1)", 0.0, 1.0, 0.6, 0.01)
        windspeed = st.slider("Wind Speed (0-1)", 0.0, 1.0, 0.2, 0.01)

    with col2:
        season = st.selectbox(
            "Season",
            [1, 2, 3, 4],
            format_func=lambda x: {1: "Spring", 2: "Summer", 3: "Fall", 4: "Winter"}[x]
        )
        weathersit = st.selectbox(
            "Weather",
            [1, 2, 3, 4],
            format_func=lambda x: {1: "Clear", 2: "Mist", 3: "Light Rain", 4: "Heavy Rain"}[x]
        )
        workingday = st.selectbox(
            "Working Day?",
            [1, 0],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )

    with col3:
        weekday = st.selectbox(
            "Day of Week",
            list(range(7)),
            format_func=lambda x: ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"][x]
        )
        mnth = st.selectbox(
            "Month",
            list(range(1, 13)),
            format_func=lambda x: [
                "Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
            ][x - 1]
        )
        yr = st.selectbox(
            "Year",
            [0, 1],
            format_func=lambda x: "2011" if x == 0 else "2012"
        )

    if st.button("Predict Bike Demand"):
        model = joblib.load(model_file)
        is_weekend = 1 if weekday in [0, 6] else 0

        input_df = pd.DataFrame([{
            "season": season,
            "yr": yr,
            "mnth": mnth,
            "hr": hr,
            "holiday": 0,
            "weekday": weekday,
            "workingday": workingday,
            "weathersit": weathersit,
            "is_weekend": is_weekend,
            "temp": temp,
            "atemp": atemp,
            "hum": hum,
            "windspeed": windspeed
        }])

        prediction = model.predict(input_df)[0]
        prediction = max(0, int(round(prediction)))

        st.success(f"Predicted Bike Demand: **{prediction} rentals**")

        if prediction < 50:
            st.info("Very low demand: few bikes needed at this hour.")
        elif prediction < 150:
            st.info("Low to moderate demand: typical off-peak hour.")
        elif prediction < 300:
            st.info("Moderate demand: normal daytime hour.")
        elif prediction < 500:
            st.info("High demand: likely a commuting peak hour.")
        else:
            st.info("Very high demand: busy rush hour or ideal conditions.")

    st.divider()
    st.subheader("Model Comparison Summary")

    comparison_df = pd.DataFrame({
        "Model": ["Linear Regression", "Random Forest"],
        "MAE": [69.22, 29.70],
        "RMSE": [97.75, 47.59],
        "R2": [0.698, 0.928]
    })

    st.dataframe(comparison_df.set_index("Model"), use_container_width=True)

    st.markdown("""
**Conclusion:** Random Forest outperforms Linear Regression across all three metrics.
It achieves an R2 of 0.928, meaning it explains 93% of the variation in hourly
bike demand, compared to 70% for Linear Regression.
""")


elif page == "Feature Importance":
    st.header("Feature Importance / Driving Variables")

    st.write("""
This page explains what drives bike rental demand. We look at Linear Regression
coefficients, Random Forest feature importance, and SHAP values to understand
which variables matter most and in what direction they affect predictions.
""")

    tab1, tab2, tab3 = st.tabs([
        "Linear Regression Coefficients",
        "Random Forest Feature Importance",
        "SHAP Explanation"
    ])

    features = [
        "season", "yr", "mnth", "hr", "holiday", "weekday",
        "workingday", "weathersit", "is_weekend",
        "temp", "atemp", "hum", "windspeed"
    ]

    df["is_weekend"] = df["weekday"].apply(lambda x: 1 if x in [0, 6] else 0)
    X_all = df[features]
    y_all = df["cnt"]

    with tab1:
        st.subheader("Linear Regression Coefficients")
        st.write("""
Coefficients show how much each feature increases or decreases predicted
rentals, holding everything else constant. Green = increases demand,
red = decreases demand.
""")

        lr_path = OUTPUT_DIR / "linear_regression_model.pkl"

        if lr_path.exists():
            lr_model = joblib.load(lr_path)

            try:
                preprocessor = lr_model.named_steps["preprocessor"]
                lr_step = lr_model.named_steps["model"]
                cat_names = preprocessor.named_transformers_["categorical"].get_feature_names_out()
                num_names = np.array(preprocessor.transformers_[1][2])
                all_feat_names = np.concatenate([cat_names, num_names])
                coef_df = pd.DataFrame({
                    "Feature": all_feat_names,
                    "Coefficient": lr_step.coef_
                }).sort_values("Coefficient", key=abs, ascending=False).head(20)
            except Exception:
                coef_df = pd.DataFrame({
                    "Feature": features,
                    "Coefficient": lr_model.coef_
                }).sort_values("Coefficient", key=abs, ascending=False)

            colors = ["#2ecc71" if c > 0 else "#e74c3c" for c in coef_df["Coefficient"]]
            fig, ax = plt.subplots(figsize=(9, 6))
            ax.barh(coef_df["Feature"][::-1], coef_df["Coefficient"][::-1], color=colors[::-1])
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_xlabel("Coefficient Value")
            ax.set_title("Top 20 Linear Regression Coefficients")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            st.markdown("""
**Interpretation:**
- **Hour (hr)** has the largest positive effect because demand rises sharply during commuting hours.
- **Temperature (atemp/temp)** increases demand because warmer conditions encourage riding.
- **Humidity (hum)** and **windspeed** reduce demand because uncomfortable conditions deter riders.
- **Working day** shows a positive effect, reflecting commuter usage patterns.
""")
        else:
            st.warning("linear_regression_model.pkl not found in model_outputs/. Run ML_Models.py first.")

    with tab2:
        st.subheader("Random Forest Feature Importance")
        st.write("""
Feature importance measures how much each variable reduces prediction error
across all trees. Higher values mean the model relied more on that feature.
""")

        image_path = OUTPUT_DIR / "random_forest_feature_importance.png"
        rf_path = OUTPUT_DIR / "random_forest_model.pkl"

        if image_path.exists():
            st.image(str(image_path), caption="Top 15 Feature Importance - Random Forest")
        elif rf_path.exists():
            rf_model = joblib.load(rf_path)
            try:
                preprocessor = rf_model.named_steps["preprocessor"]
                rf_step = rf_model.named_steps["model"]
                cat_names = preprocessor.named_transformers_["categorical"].get_feature_names_out()
                num_names = np.array(preprocessor.transformers_[1][2])
                all_feat_names = list(np.concatenate([cat_names, num_names]))
            except Exception:
                rf_step = rf_model
                all_feat_names = features

            importance_df = pd.DataFrame({
                "Feature": all_feat_names,
                "Importance": rf_step.feature_importances_
            }).sort_values("Importance", ascending=False).head(15)

            fig, ax = plt.subplots(figsize=(9, 6))
            ax.barh(
                importance_df["Feature"][::-1],
                importance_df["Importance"][::-1],
                color="#3498db"
            )
            ax.set_xlabel("Importance Score")
            ax.set_title("Top 15 Feature Importance - Random Forest")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        else:
            st.warning("Feature importance image not found. Run ML_Models.py first.")

        st.markdown("""
**Interpretation:**
- **Hour of day (hr)** is the strongest predictor because demand spikes at 8 AM and 5-6 PM.
- **Apparent temperature (atemp)** is second most important because rider comfort drives usage.
- **Humidity (hum)** reduces demand significantly at high values.
- **Working day (workingday)** separates commuter weekday peaks from leisure weekend patterns.
- **Year (yr)** captures overall growth in bike sharing adoption from 2011 to 2012.
""")

    with tab3:
        st.subheader("SHAP Values - Individual Prediction Explanation")
        st.write("""
SHAP shows how much each feature pushed a specific prediction higher or lower
compared to the average. Select a row to explain that individual prediction.
""")

        rf_path = OUTPUT_DIR / "random_forest_model.pkl"

        if rf_path.exists():
            try:
                import shap

                rf_model = joblib.load(rf_path)

                idx = st.number_input(
                    "Select a row to explain (0 = first row)",
                    min_value=0,
                    max_value=len(df) - 1,
                    value=0,
                    step=1
                )

                if st.button("Run SHAP Explanation"):
                    with st.spinner("Computing SHAP values..."):
                        try:
                            preprocessor = rf_model.named_steps["preprocessor"]
                            rf_step = rf_model.named_steps["model"]
                            cat_names = preprocessor.named_transformers_["categorical"].get_feature_names_out()
                            num_names = np.array(preprocessor.transformers_[1][2])
                            all_feat_names = list(np.concatenate([cat_names, num_names]))
                            X_single = preprocessor.transform(X_all.iloc[[idx]])
                        except Exception:
                            rf_step = rf_model
                            all_feat_names = features
                            X_single = X_all.iloc[[idx]].values

                        explainer = shap.TreeExplainer(rf_step)
                        shap_vals_single = explainer.shap_values(X_single)

                        predicted = max(0, int(round(rf_model.predict(X_all.iloc[[idx]])[0])))
                        actual = int(y_all.iloc[idx])

                        col_a, col_b = st.columns(2)
                        col_a.metric("Actual Rentals", actual)
                        col_b.metric("Predicted Rentals", predicted)

                        shap_df = pd.DataFrame({
                            "Feature": all_feat_names,
                            "SHAP Value": shap_vals_single[0]
                        }).sort_values("SHAP Value", key=abs, ascending=False).head(15)

                        colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in shap_df["SHAP Value"]]
                        fig, ax = plt.subplots(figsize=(9, 5))
                        ax.barh(shap_df["Feature"][::-1], shap_df["SHAP Value"][::-1], color=colors[::-1])
                        ax.axvline(0, color="black", linewidth=0.8)
                        ax.set_xlabel("SHAP Value (impact on this prediction)")
                        ax.set_title(f"Feature Contributions for Row {idx}")
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()

                        st.markdown("""
**Green bars** pushed this prediction **higher** than average.
**Red bars** pushed this prediction **lower** than average.
""")
            except ImportError:
                st.warning("SHAP is not installed. Run `pip install shap` to enable this section.")
        else:
            st.warning("random_forest_model.pkl not found in model_outputs/. Run ML_Models.py first.")


elif page == "Model Comparison / Tuning":
    st.header("Model Comparison and Hyperparameter Tuning")

    st.write("""
This page tunes the Random Forest by testing different combinations of
hyperparameters and compares the results against the baseline models.
""")

    features = [
        "season", "yr", "mnth", "hr", "holiday", "weekday",
        "workingday", "weathersit", "is_weekend",
        "temp", "atemp", "hum", "windspeed"
    ]

    df["is_weekend"] = df["weekday"].apply(lambda x: 1 if x in [0, 6] else 0)
    X_all = df[features]
    y_all = df["cnt"]

    lr_path = OUTPUT_DIR / "linear_regression_model.pkl"
    rf_path = OUTPUT_DIR / "random_forest_model.pkl"

    if not lr_path.exists() or not rf_path.exists():
        st.warning("Model files not found in model_outputs/. Run ML_Models.py first.")
    else:
        lr_model = joblib.load(lr_path)
        rf_model = joblib.load(rf_path)

        def get_metrics(model, X, y):
            preds = np.maximum(model.predict(X), 0)
            return {
                "MAE": round(mean_absolute_error(y, preds), 3),
                "RMSE": round(float(np.sqrt(mean_squared_error(y, preds))), 3),
                "R2": round(r2_score(y, preds), 4),
            }

        lr_metrics = get_metrics(lr_model, X_all, y_all)
        rf_metrics = get_metrics(rf_model, X_all, y_all)

        st.subheader("Baseline Model Performance")
        baseline_df = pd.DataFrame([
            {"Model": "Linear Regression", **lr_metrics},
            {"Model": "Random Forest (baseline)", **rf_metrics},
        ])
        st.table(baseline_df)

        final_path = OUTPUT_DIR / "final_model_comparison.csv"
        if final_path.exists():
            st.subheader("Previously Saved Tuning Results")
            st.dataframe(pd.read_csv(final_path), use_container_width=True)

        st.divider()
        st.subheader("Run Your Own Hyperparameter Search")

        col1, col2 = st.columns(2)

        with col1:
            n_est_opts = st.multiselect(
                "n_estimators",
                options=[50, 100, 200, 300],
                default=[100, 200]
            )
            depth_opts = st.multiselect(
                "max_depth",
                options=[10, 15, 20, 25],
                default=[15, 20]
            )

        with col2:
            split_opts = st.multiselect(
                "min_samples_split",
                options=[2, 5, 10],
                default=[2, 5]
            )
            leaf_opts = st.multiselect(
                "min_samples_leaf",
                options=[1, 2, 4],
                default=[1, 2]
            )

        if not all([n_est_opts, depth_opts, split_opts, leaf_opts]):
            st.warning("Please select at least one value for each parameter.")
        else:
            param_grid = list(ParameterGrid({
                "n_estimators": n_est_opts,
                "max_depth": depth_opts,
                "min_samples_split": split_opts,
                "min_samples_leaf": leaf_opts,
            }))
            st.info(f"This will run **{len(param_grid)} experiments**.")

            if st.button("Run Tuning"):
                from sklearn.model_selection import train_test_split

                X_train, X_test, y_train, y_test = train_test_split(
                    X_all,
                    y_all,
                    test_size=0.2,
                    random_state=42
                )

                try:
                    preprocessor = rf_model.named_steps["preprocessor"]
                    X_train_t = preprocessor.transform(X_train)
                    X_test_t = preprocessor.transform(X_test)
                except Exception:
                    X_train_t, X_test_t = X_train, X_test

                results = []
                progress = st.progress(0)
                status = st.empty()

                for i, params in enumerate(param_grid):
                    status.text(f"Experiment {i + 1}/{len(param_grid)}: {params}")
                    m = RandomForestRegressor(**params, random_state=42, n_jobs=-1)
                    m.fit(X_train_t, y_train)
                    results.append({
                        "Experiment": i + 1,
                        **params,
                        **get_metrics(m, X_test_t, y_test)
                    })
                    progress.progress((i + 1) / len(param_grid))

                status.text("Done!")
                results_df = pd.DataFrame(results)
                results_df.to_csv(OUTPUT_DIR / "tuning_results.csv", index=False)

                st.subheader("All Experiments (sorted by MAE)")
                st.dataframe(results_df.sort_values("MAE"), use_container_width=True)

                best = results_df.loc[results_df["MAE"].idxmin()]

                st.subheader("Best Configuration Found")
                c1, c2, c3 = st.columns(3)
                c1.metric("Best MAE", best["MAE"])
                c2.metric("Best RMSE", best["RMSE"])
                c3.metric("Best R2", best["R2"])

                st.json({
                    "n_estimators": int(best["n_estimators"]),
                    "max_depth": int(best["max_depth"]),
                    "min_samples_split": int(best["min_samples_split"]),
                    "min_samples_leaf": int(best["min_samples_leaf"])
                })

                final_df = pd.DataFrame([
                    {"Model": "Linear Regression", **lr_metrics},
                    {"Model": "Random Forest (baseline)", **rf_metrics},
                    {
                        "Model": "Random Forest (tuned)",
                        "MAE": best["MAE"],
                        "RMSE": best["RMSE"],
                        "R2": best["R2"]
                    },
                ])
                final_df.to_csv(OUTPUT_DIR / "final_model_comparison.csv", index=False)

                st.subheader("Final Model Comparison")
                st.table(final_df)

                fig, ax = plt.subplots(figsize=(8, 4))
                ax.bar(
                    final_df["Model"],
                    final_df["MAE"],
                    color=["#e74c3c", "#3498db", "#2ecc71"]
                )
                ax.set_ylabel("MAE (lower is better)")
                ax.set_title("Mean Absolute Error by Model")
                plt.xticks(rotation=15, ha="right")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

                fig2, ax2 = plt.subplots(figsize=(10, 4))
                ax2.plot(results_df["Experiment"], results_df["MAE"], marker="o", color="#3498db")
                ax2.axhline(
                    rf_metrics["MAE"],
                    color="#e74c3c",
                    linestyle="--",
                    label="Baseline Random Forest"
                )
                ax2.set_xlabel("Experiment")
                ax2.set_ylabel("MAE")
                ax2.set_title("MAE Per Tuning Experiment")
                ax2.legend()
                plt.tight_layout()
                st.pyplot(fig2)
                plt.close()

                st.markdown("""
**How to interpret:** If the tuned MAE is lower than the baseline, use the tuned model.
If they are similar, keep the original Random Forest. Share `final_model_comparison.csv`
with Person 5 for the conclusion page.
""")


elif page == "Conclusion":
    st.header("Conclusion")

    st.write("""
The Random Forest Regressor was the best-performing model. It achieved the
lowest prediction error and the highest R2 score, meaning it explained most of
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
