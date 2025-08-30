import streamlit as st
import requests
import os

st.set_page_config(page_title="Stock Research Agent", layout="wide")

st.title("📊 Stock Research Agent")

# --- Sidebar ---
st.sidebar.header("Upload Annual Report")
uploaded_file = st.sidebar.file_uploader("Upload PDF", type=["pdf"])
file_name = None
if uploaded_file:
    file_name = os.path.splitext(uploaded_file.name)[0]

    # Create a reports folder if not exists
    save_dir = "uploads/reports"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{file_name}.pdf")

    # Save file locally
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.sidebar.success("✅ File uploaded!")

    # Call API to build vectorstore
    vectorstore_path = f"annual-report-agent/vectorstore/{file_name}"
    if not os.path.exists(vectorstore_path):
        with st.spinner("Processing and building vectorstore..."):
            try:
                resp = requests.post(
                    "http://localhost:8000/build_vectorstore",
                    json={"file_path": save_path, "file_name": file_name},
                )
                if resp.status_code == 200:
                    st.success("Vectorstore saved")
                else:
                    st.error("Failed to build vectorstore")
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.info("Vectorstore already exists. Skipping rebuild.")

    # Call API to generate analysis
    if st.sidebar.button("Generate Analysis"):
        with st.spinner("Processing report..."):
            try:
                resp = requests.post(
                    "http://localhost:8000/generate_report",
                    json={"company_name": file_name},
                )
                if resp.status_code == 200:
                    st.session_state["analysis"] = {
                        "report": resp.json().get("report", "No report generated.")
                    }
                    st.sidebar.success("✅ Analysis generated!")
                else:
                    st.error("Failed to generate analysis")
            except Exception as e:
                st.error(f"Error: {e}")

# --- Main Tabs ---
tabs = st.tabs(["📈 Research Report", "💬 Chat"])

# Research Report Tab
with tabs[0]:
    st.header("Equity Research-Style Report")
    if "analysis" in st.session_state:
        st.markdown(st.session_state["analysis"]["report"])
    else:
        st.info("Upload a report and generate analysis to view here.")

# Chat Tab
with tabs[1]:
    st.header("Chat with the Report")
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    user_input = st.text_input("Ask a question about the report:", key="user_input")
    if st.button("Send"):
        if user_input:
            st.session_state["chat_history"].append({"role": "user", "content": user_input})
            try:
                resp = requests.post(
                    "http://localhost:8000/chat",
                    json={"query": user_input, "file_name": file_name},
                )
                if resp.status_code == 200:
                    response_text = resp.json().get("answer", "No response from assistant.")
                else:
                    response_text = "Error: Failed to get response."
            except Exception as e:
                response_text = f"Error: {e}"

            st.session_state["chat_history"].append({"role": "assistant", "content": response_text})

    # Display chat history
    for msg in st.session_state["chat_history"]:
        if msg["role"] == "user":
            st.markdown(f"**You:** {msg['content']}")
        else:
            st.markdown(f"**Assistant:** {msg['content']}")
