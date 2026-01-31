import os
import pandas as pd
import streamlit as st
import altair as alt

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="🚦 Intelligent Traffic Dashboard",
    page_icon="🚦",
    layout="wide"
)

st.title("🚦 Intelligent Traffic Management Dashboard")
st.caption("Queue Analysis & Overspeed Evidence System")

# =========================
# FIXED PROJECT PATH
# =========================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

CSV_PATH = os.path.join(PROJECT_ROOT, "outputs", "finalspeedlog.csv")
VIDEO_PATH = os.path.join(PROJECT_ROOT, "outputs", "finaloutputvideo.mp4")
VIOLATION_DIR = os.path.join(PROJECT_ROOT, "violations", "overspeed")

# =========================
# LOAD CSV
# =========================
if not os.path.exists(CSV_PATH):
    st.error("❌ finalspeedlog.csv not found. Run main.py first.")
    st.stop()

df = pd.read_csv(CSV_PATH)

# =========================
# LATEST METRICS
# =========================
latest = df.iloc[-1]

c1, c2, c3, c4 = st.columns(4)
c1.metric("🚗 Total Vehicles", int(latest["Total_Vehicle_Count"]))
c2.metric("🚧 Queue Count", int(latest["Queue_Count"]))
c3.metric("📏 Queue Length (m)", f"{latest['Queue_Length']:.2f}")
c4.metric("📊 Queue Density", f"{latest['Queue_Density']:.2f}")

st.divider()

# =========================
# FRAME RANGE FILTER
# =========================
st.subheader("🎞️ Traffic Data Explorer")

frame_range = st.slider(
    "Select Frame Range",
    int(df["Frame"].min()),
    int(df["Frame"].max()),
    (int(df["Frame"].min()), int(df["Frame"].max()))
)

filtered_df = df[
    (df["Frame"] >= frame_range[0]) &
    (df["Frame"] <= frame_range[1])
]

st.dataframe(filtered_df.tail(100), use_container_width=True)

st.divider()

# =========================
# GRAPHS
# =========================
st.subheader("📈 Traffic Trends")

chart_df = filtered_df.tail(300)

st.altair_chart(
    alt.Chart(chart_df).mark_line(color="red").encode(
        x="Frame", y="Queue_Count"
    ).properties(title="Queue Count"),
    use_container_width=True
)

st.altair_chart(
    alt.Chart(chart_df).mark_line(color="orange").encode(
        x="Frame", y="Queue_Length"
    ).properties(title="Queue Length (m)"),
    use_container_width=True
)

st.altair_chart(
    alt.Chart(chart_df).mark_line(color="green").encode(
        x="Frame", y="Queue_Density"
    ).properties(title="Queue Density"),
    use_container_width=True
)

# =========================
# OVERSPEED GALLERY
# =========================
st.divider()
st.subheader("🚨 Overspeed Vehicle Evidence")

if os.path.exists(VIOLATION_DIR):
    images = sorted(os.listdir(VIOLATION_DIR))
    if images:
        cols = st.columns(4)
        for i, img in enumerate(images):
            with cols[i % 4]:
                st.image(
                    os.path.join(VIOLATION_DIR, img),
                    caption=img,
                    use_container_width=True
                )
    else:
        st.info("No overspeed violations detected.")
else:
    st.warning("Violation folder not found.")

# =========================
# VIDEO PREVIEW
# =========================
st.divider()
