import streamlit as st
import requests
import pandas as pd

# URL of your deployed FastAPI backend
API_URL = "https://smartattendancesystem-production-e0b7.up.railway.app"

# Admin key input
ADMIN_KEY = st.text_input("Enter Admin Key", type="password")

if not ADMIN_KEY:
    st.warning("Please enter the admin key to continue.")
    st.stop()

headers = {"Authorization": f"Bearer {ADMIN_KEY}"}

# Fetch attendance data
with st.spinner("Fetching attendance data..."):
    try:
        r = requests.get(f"{API_URL}/admin/attendance-records", headers=headers)
        r.raise_for_status()
        df = pd.DataFrame(r.json())
        if df.empty:
            st.info("No attendance records found.")
    except Exception as e:
        st.error(f"Failed to fetch data: {e}")
        st.stop()

st.title("📊 Attendance Admin Panel")

# Show table
st.subheader("Attendance Records")
st.dataframe(df)

# Fetch summary
with st.spinner("Fetching summary stats..."):
    try:
        summary_r = requests.get(f"{API_URL}/admin/attendance_summary", headers=headers)
        summary_r.raise_for_status()
        summary = summary_r.json()
    except Exception as e:
        st.error(f"Failed to fetch summary: {e}")
        st.stop()

st.subheader("Summary Stats")
st.write(f"Total Present: {summary.get('total_present', 0)}")
st.write(f"Total Absent: {summary.get('total_absent', 0)}")

# Bar chart by student
st.subheader("Attendance by Student")
by_student_df = pd.DataFrame(list(summary.get("by_student", {}).items()), columns=["Student", "Present Count"])
if not by_student_df.empty:
    st.bar_chart(by_student_df.set_index("Student"))
else:
    st.info("No student data to display.")

# Optional: Export CSV
csv = df.to_csv(index=False).encode("utf-8")
st.download_button("Download CSV", csv, "attendance_report.csv", "text/csv")
