import sys
sys.path.append("/mnt/data")


import streamlit as st
import sys
from io import BytesIO, StringIO
from contextlib import redirect_stdout

# ensure modules are in path
sys.path.append("/mnt/data")

from helper import print_explanation_summary, analyze_and_display
from keyphrase_analysis import run_pipeline as kp_run
from span_analysis import run_pipeline as span_run

st.set_page_config(page_title="SDG Analyzer", layout="wide")
st.title("SDG Classification & Explainability")

# Sidebar controls
st.sidebar.header("Upload & Settings")
mode = st.sidebar.selectbox("Mode", ["Keyphrase", "IG Span"])
uploaded = st.sidebar.file_uploader("Upload PDF", type=["pdf"])
max_sents = st.sidebar.slider("Max Sentences", 10, 200, 50)

# Parameter toggles
if mode == "Keyphrase":
    st.sidebar.subheader("Keyphrase Parameters")
    top_k_kp = st.sidebar.slider("Top K Keyphrases", 1, 20, 5)
    agree_thresh = st.sidebar.slider("Agreement Threshold", 0.0, 1.0, 0.1, format="%.2f")
    disagree_thresh = st.sidebar.slider("Disagreement Threshold", 0.0, 1.0, 0.2, format="%.2f")
    min_conf = st.sidebar.slider("Minimum Confidence", 0.0, 1.0, 0.5, format="%.2f")
else:
    st.sidebar.subheader("IG Span Parameters")
    max_span_len = st.sidebar.slider("Max Span Length", 1, 20, 10)
    top_k_sp = st.sidebar.slider("Top K Spans", 1, 20, 5)

# Initialize session state for results
if 'results' not in st.session_state:
    st.session_state['results'] = None

# Run analysis when button clicked
if uploaded:
    pdf_bytes = uploaded.read()
    tmp_path = "/tmp/uploaded.pdf"
    with open(tmp_path, "wb") as f:
        f.write(pdf_bytes)

    if st.sidebar.button("🚀 Run Analysis"):
        if mode == "Keyphrase":
            res = kp_run(
                pdf_path=tmp_path,
                max_sentences=max_sents,
                top_k=top_k_kp,
                agree_thresh=agree_thresh,
                disagree_thresh=disagree_thresh,
                min_conf=min_conf
            )
        else:
            res = span_run(
                pdf_path=tmp_path,
                max_sentences=max_sents,
                max_span_len=max_span_len
            )
        st.session_state['results'] = res

# Display results if available
if st.session_state['results'] is not None:
    results = st.session_state['results']

    # Capture and display the summary
    buf = StringIO()
    with redirect_stdout(buf):
        print_explanation_summary(results)
    summary_text = buf.getvalue()

    st.subheader("📊 Pipeline Summary")
    st.text_area("Summary & Results", summary_text, height=300)

    # Show any matplotlib figures
    import matplotlib.pyplot as plt
    for num in plt.get_fignums():
        fig = plt.figure(num)
        st.pyplot(fig)

    # Interactive sentence-level details
    idx = st.selectbox(
        "Select Sentence Index",
        list(range(len(results.get("results", [])))),
        format_func=lambda i: results["results"][i]["sentence"][ :80]
    )
    detail_buf = StringIO()
    with redirect_stdout(detail_buf):
        analyze_and_display(idx, results)
    detail_text = detail_buf.getvalue()

    st.subheader(f"📝 Details for Sentence {idx}")
    st.text_area("Detailed Analysis", detail_text, height=400)
else:
    if uploaded:
        st.info("Click ‘Run Analysis’ to see results.")
    else:
        st.info("Please upload a PDF file to begin analysis.")


