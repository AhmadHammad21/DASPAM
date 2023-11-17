import streamlit as st
import pandas as pd
import json
from utils import preprocess_df
import pickle
import numpy as np
import warnings

warnings.filterwarnings("ignore")


# Load the pipeline
with open("./artifacts/model.pkl", "rb") as file:
    loaded_pipeline = pickle.load(file)


ingredients_table = pd.read_csv("./data/processed/Ingredients_Weights.csv")

st.set_page_config(page_title="DemandGuru")
st.title("DemandGuru")
st.write(
    "Welcome to ***DemandGuru*** your personal Demand and Supply predictive Algorithmic Model"
)
# Display the image with Streamlit
st.image("components/res.jpg", use_column_width=True, output_format="auto")

# Center the image by adjusting the layout
col_width = st.get_option("deprecation.showfileUploaderEncoding") - 100
st.markdown(
    f"""
    <style>
        .reportview-container .main .block-container{{
            max-width: {col_width}px;
            padding-top: 0px;
            padding-right: 0px;
            padding-left: 0px;
            padding-bottom: 0px;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.caption(
    "Our app aims to help resturants predict their products demand over a period of time"
)

data_file = st.file_uploader("Upload data orders file", type=("csv", "xlsx"))
if data_file is not None:  # this is used to check if the file was actually uploaded
    file_extension = data_file.name.split(".")[-1].lower()

    if file_extension == "csv":
        data_frame = pd.read_csv(data_file)
    elif file_extension in ["xls", "xlsx"]:
        data_frame = pd.read_excel(data_file)
    else:
        st.error("Unsupported file format. Please upload a CSV or Excel file.")
        data_frame = None

    if data_frame is not None:
        st.write(data_frame.head())

    # ingrediant_cost = data_frame.set_index('pizza_name')['clean_pizza_id'].to_dict()

    st.write("### Click to Show Next Week Ingrediants Demand For your Restaurent")
    continue_butt = st.button(label="Show Results")
    if continue_butt:
        last_date_found = "2015-12-14"  # data_frame['Week_Start'].max()

        print("Last Date found", last_date_found)

        latest_week_df = data_frame[data_frame["Week_Start"] == last_date_found]

        # transforming it to proper form
        processed_data = preprocess_df(latest_week_df)
        print("processed data", processed_data)

        predictions = np.round(loaded_pipeline.predict(processed_data))

        predictions_df = pd.DataFrame(
            {"Pizza ID": processed_data["pizza_id"], "Predicted Orders": predictions}
        )

        predictions_df = predictions_df.sort_values("Predicted Orders", ascending=False)

        # Display DataFrames side by side
        col1, col2 = st.columns([1, 2])

        with col1:
            st.dataframe(predictions_df.set_index(predictions_df.columns[0]))

        with col2:
            st.dataframe(ingredients_table.set_index(ingredients_table.columns[0]))
