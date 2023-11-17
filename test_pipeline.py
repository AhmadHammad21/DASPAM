import joblib 
import pandas as pd
import numpy as np
from utils import preprocess_df


# load data
data = pd.read_csv("data\processed\demo_data.csv")


pizza_name = "bbq_ckn"
pizza_size = "Small"

sizes_dict = {"Large": "l", "Medium": "m", "Small": "s"}

query_key = pizza_name + "_" + sizes_dict[pizza_size]

pizza_data = data[data['pizza_id'] == query_key]

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


# Load the pipeline
loaded_pipeline = joblib.load('./artifacts/model_pipeline.joblib')

predictions = np.round(loaded_pipeline.predict(processed_data))
print(predictions)


predictions_df = pd.DataFrame({'Week': [i for i in range(1, 13)],
                               "Date": weekly_dates, 
                               "Predicted Orders": predictions})

print(predictions_df)

