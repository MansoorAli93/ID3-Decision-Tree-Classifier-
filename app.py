import streamlit as st
import pandas as pd
import numpy as np
import math

st.set_page_config(page_title="ID3 Decision Tree Classifier", layout="centered")
st.title("ID3 Decision Tree Classifier")
st.write("Train and test a simple **ID3 Decision Tree** using categorical data.")

def entropy(col):
    values, counts = np.unique(col, return_counts=True)
    return -sum((c / len(col)) * math.log2(c / len(col)) for c in counts)

def info_gain(df, attr, target):
    total_entropy = entropy(df[target])
    vals = df[attr].unique()
    weighted_entropy = sum(
        (len(df[df[attr] == v]) / len(df)) * entropy(df[df[attr] == v][target])
        for v in vals
    )
    return total_entropy - weighted_entropy

def id3(df, target, attrs):
    if len(df[target].unique()) == 1:
        return df[target].iloc[0]
    if not attrs:
        return df[target].mode()[0]

    best = max(attrs, key=lambda a: info_gain(df, a, target))
    tree = {best: {}}

    for val in df[best].unique():
        sub_df = df[df[best] == val]
        tree[best][val] = id3(sub_df, target, [a for a in attrs if a != best])
    return tree

def predict(tree, input_data):
    if not isinstance(tree, dict):
        return tree
    root = next(iter(tree))
    val = input_data.get(root)
    if val in tree[root]:
        return predict(tree[root][val], input_data)
    return "Unknown"

data_dict = {
    "outlook": ["sunny","sunny","overcast","rain","rain","overcast","sunny","sunny","overcast","rain","overcast","overcast","rain","sunny"],
    "humidity": ["high","normal","high","normal","high","high","normal","normal","normal","normal","normal","high","high","normal"],
    "playtennis": ["no","yes","yes","yes","no","yes","yes","yes","no","yes","no","yes","yes","yes"]
}

df = pd.DataFrame(data_dict)

st.sidebar.header("Settings")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type="csv")

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success("CSV Loaded Successfully")
    except:
        st.sidebar.error("Invalid CSV file")

show_data = st.sidebar.checkbox("Show Dataset Preview", value=True)
if show_data:
    st.subheader("Dataset Preview")
    st.dataframe(df, use_container_width=True)

st.subheader("Model Configuration")
target_col = st.selectbox("Select Target Column", df.columns, index=len(df.columns) - 1)
features = [c for c in df.columns if c != target_col]

if st.button("Train Model"):
    tree = id3(df, target_col, features)
    st.session_state["tree"] = tree
    st.success("Model Trained Successfully!")
    st.json(tree)

if "tree" in st.session_state:
    st.subheader("Make Prediction")
    inputs = {}
    cols = st.columns(len(features))
    for i, col in enumerate(features):
        with cols[i]:
            inputs[col] = st.selectbox(col, df[col].unique())

    if st.button("Predict"):
        result = predict(st.session_state["tree"], inputs)
        if result.lower() in ["yes", "true", "1"]:
            st.success(f"Prediction: **{result}**")
        elif result.lower() in ["no", "false", "0"]:
            st.error(f"Prediction: **{result}**")
        else:
            st.info(f"Prediction: **{result}**")

st.markdown("---")
st.caption("Built with Streamlit • ID3 Decision Tree")




