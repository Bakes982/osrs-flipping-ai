import streamlit as st
import pandas as pd
import re
import plotly.express as px

st.set_page_config(page_title="OSRS Bot Tracker", layout="wide")

# Title and File Loading
st.title("📈 OSRS Flipping Dashboard")
LOG_FILE = "logfile-1770495016916.log"

def parse_logs(file_path):
    data = []
    # Regex to find: Timestamp, Item Name, and Profit
    pattern = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \[INFO\] FLIP COMPLETE: (.+?) \| Profit: (-?\d+)")
    
    try:
        with open(file_path, "r") as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    timestamp, item, profit = match.groups()
                    data.append({
                        "Timestamp": pd.to_datetime(timestamp),
                        "Item": item,
                        "Profit": int(profit)
                    })
        return pd.DataFrame(data)
    except FileNotFoundError:
        return None

df = parse_logs(LOG_FILE)

if df is not None and not df.empty:
    # Calculations
    df = df.sort_values("Timestamp")
    df["Cumulative Profit"] = df["Profit"].cumsum()
    total_profit = df["Profit"].sum()
    total_flips = len(df)
    
    # Header Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Profit/Loss", f"{total_profit:,} gp", delta=int(total_profit))
    col2.metric("Total Flips", total_flips)
    col3.metric("Avg. Profit per Flip", f"{int(total_profit/total_flips)} gp")

    # Profit Over Time Chart
    st.subheader("Performance Over Time")
    fig = px.line(df, x="Timestamp", y="Cumulative Profit", title="Cumulative GP Growth")
    st.plotly_chart(fig, use_container_width=True)

    # Raw Data Table
    st.subheader("Recent Flips")
    st.dataframe(df.sort_values("Timestamp", ascending=False), use_container_width=True)
else:
    st.error(f"Could not find {LOG_FILE} or no flips were found in the log.")
    st.info("Make sure the log file is in the same folder as this script!")