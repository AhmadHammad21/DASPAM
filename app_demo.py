import streamlit as st
import pandas as pd
import json
from utils import preprocess_df
import pickle
import numpy as np
import warnings 
warnings.filterwarnings("ignore")

# variables
data_file = 0
data_frame = 0
pizza_type = 0
pizza_size = 0


# Load the JSON file
with open("./artifacts/pizza_dict.json", 'r') as json_file:
    pizza_dict = json.load(json_file)

# Load the JSON file
with open("./artifacts/pizza_names.json", 'r') as json_file:
    pizza_names = json.load(json_file)

    # Load the pipeline
with open('./artifacts/model.pkl', 'rb') as file:
    loaded_pipeline = pickle.load(file)

sizes_dict = {"Large": "l", "Medium": "m", "Small": "s", 
                  "L": "l", "M": "m", "S": "s"}


st.set_page_config(page_title="DASPAM")
st.title("DASPAM")
st.write(
    "welcome to ***DASPAM*** your personal Demand and Supply predictive Algrathmic Model"
)
st.image(image="components/res.jpg")
st.caption(
    "our app aims to help resturants predict thier products demand over a period of time"
)

data_file = st.file_uploader("Upload a file", type=("csv"))
if data_file is not None:  # this is used to check if the file was actually uploaded
    data_frame = pd.read_csv(data_file)
    st.write(data_frame.head())

    pizza_names_list = list(pizza_names.keys())
    pizza_type = st.selectbox(
        label="Select the Type of pizza",
        options=pizza_names_list,
    )

    single_pizza_type_options = ['Large', 'Medium', 'Small'] #pizza_dict[pizza_type]
    pizza_size = st.radio(
        label="Select pizza size", options=single_pizza_type_options, index=2
    )
    print("pizza_size", pizza_size, type(pizza_size))

    st.write(f"Predict Next 12 Weeks For {pizza_type} {pizza_size}")
    continue_butt = st.button(label="Predict")
    if continue_butt:  # this checks if the pizza_size and pizza_type are as selected by user
      
        query_key = pizza_names[pizza_type] + "_" + sizes_dict[pizza_size.strip()]
        print("query key", query_key)
        pizza_data = data_frame[data_frame['pizza_id'] == query_key]

        last_date_found = pizza_data['Week_Start'].max()

        print("Last Date found", last_date_found)

        # Convert the starting date to a datetime object
        start_date = pd.to_datetime(last_date_found, format='%Y-%m-%d')

        # Generate 12 weekly dates
        weekly_dates = pd.date_range(start=start_date, periods=12, freq='W')
        print("weekly dates", weekly_dates)
        print("found data", pizza_data)

        # transforming it to proper form
        processed_data = preprocess_df(pizza_data)
        print("processed data", processed_data)

        predictions = np.round(loaded_pipeline.predict(processed_data))
        print(predictions)


        predictions_df = pd.DataFrame({'Week': [i for i in range(1, 13)],
                                    "Date": weekly_dates, 
                                    "Predicted Orders": predictions})

        predictions_df['Date'] =predictions_df['Date'].dt.date

        st.write(predictions_df)