"""
Research Gap Discovery Engine - Streamlit Dashboard.
AI-powered research analysis platform for discovering research gaps.
"""

import json

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.dashboard_utils import (
    format_limitation,
    get_analytics_data,
    get_cluster_analysis,
    get_limitations_by_paper,
    get_research_gaps_csv,
    get_statistics,
    load_all_limitations,
    load_research_gaps,
    perform_semantic_search,
    truncate_text,
)
from src.pdf_report import generate_pdf_report
from src.literature_review import generate_literature_review
from src.upload_handler import handle_pdf_upload
from pathlib import Path


st.set_page_config(
    page_title="Research Gap Discovery Engine",
    page_icon="RG",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": (
            "Research Gap Discovery Engine v1.0\n"
            "A professional platform for identifying research opportunities."
        )
    },
)

st.markdown(
    """
<style>
body {
    font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
}

.hero-section {
    background: linear-gradient(135deg, #1f4d78 0%, #2a7f62 100%);
    padding: 36px 24px;
    border-radius: 16px;
    color: white;
    text-align: center;
    margin-bottom: 28px;
}

.hero-section h1 {
    margin: 0;
    font-size: 2.4rem;
    font-weight: 700;
}

.hero-section p {
    margin: 10px 0 0 0;
    font-size: 1rem;
    opacity: 0.92;
}

.metric-card {
    background: linear-gradient(160deg, #f8fbff 0%, #eef5fb 100%);
    border: 1px solid #d9e5f2;
    border-radius: 14px;
    padding: 18px;
    margin-bottom: 14px;
}

.metric-label {
    color: #4a6178;
    font-size: 0.85rem;
    margin-bottom: 8px;
}

.metric-value {
    color: #15344d;
    font-size: 1.9rem;
    font-weight: 700;
}

.section-note {
    color: #557085;
    font-size: 0.95rem;
}
</style>
""",
    unsafe_allow_html=True,
)


def render_metric_card(label: str, value: int, caption: str | None = None):
    caption_html = (
        f'<div class="section-note" style="margin-top:6px;">{caption}</div>'
        if caption
        else ""
    )
    st.markdown(
        f"""
<div class="metric-card">
    <div class="metric-label">{label}</div>
    <div class="metric-value">{value}</div>
    {caption_html}
</div>
""",
        unsafe_allow_html=True,
    )


with st.sidebar:
    st.title("Research Gap Discovery")
    st.caption("Analysis Platform")

    page = st.radio(
        "Navigation",
        ["Home", "Upload PDF", "Paper Search", "Limitations", "Clusters", "Research Gaps", "Analytics", "Reports"],
        label_visibility="collapsed",
    )

    st.divider()

    stats = get_statistics()
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Papers", stats["total_papers"])
        st.metric("Gaps", stats["total_research_gaps"])
    with col2:
        st.metric("Limitations", stats["total_limitations"])
        st.metric("Clusters", stats["total_clusters"])

    st.divider()
    st.caption("Features")
    st.write("- Semantic search with FAISS")
    st.write("- RAG with Gemini API")
    st.write("- KMeans clustering")
    st.write("- AI-generated insights")
    st.write("- Exportable results")


def page_home():
    st.markdown(
        """
    <div class="hero-section">
        <h1>Research Gap Discovery Engine</h1>
        <p>Intelligent analysis of research limitations to identify emerging opportunities</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("## Project Overview")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
### What It Does

- Loads multiple research papers
- Extracts methods, datasets, and limitations
- Clusters related limitations into themes
- Generates research gaps and potential ideas
- Supports semantic search with grounded answers
"""
        )

    with col2:
        st.markdown(
            """
### Technology Stack

- Python, Streamlit, Plotly
- LangChain and PyPDF
- Sentence Transformers and FAISS
- Scikit-learn for clustering
- Gemini API for RAG and gap generation
"""
        )

    st.divider()
    st.markdown("## Platform Statistics")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_metric_card("Research Papers", stats["total_papers"])
    with col2:
        render_metric_card("Limitations Found", stats["total_limitations"])
    with col3:
        render_metric_card("Research Themes", stats["total_clusters"])
    with col4:
        render_metric_card("Research Gaps", stats["total_research_gaps"])

    st.divider()
    st.markdown("## Quick Navigation")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """
### Paper Search

Use semantic retrieval to ask natural-language questions across the paper set.
"""
        )
    with col2:
        st.markdown(
            """
### Limitations Explorer

Browse extracted limitations by paper and filter them by topic.
"""
        )
    with col3:
        st.markdown(
            """
### Research Gaps

Review clustered themes, generated gaps, supporting evidence, and exportable results.
"""
        )


def page_upload():
    st.title("📤 Upload Research Papers")
    st.write("Upload PDF files to add new research papers to the analysis engine.")
    st.divider()

    st.markdown("### Upload Files")
    st.info(
        "✨ **New PDFs Only**: Already processed papers are automatically skipped. "
        "Embeddings and FAISS index are updated incrementally for new papers."
    )

    # File uploader
    uploaded_files = st.file_uploader(
        "Select PDF files to upload",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if not uploaded_files:
        st.markdown("### Instructions")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                """
**Step 1: Upload PDFs**
- Click to select one or more PDF files
- Supports multiple simultaneous uploads
- Files are saved to `data/papers/`
"""
            )
        with col2:
            st.markdown(
                """
**Step 2: Processing**
- Extracts methods, datasets, and limitations
- Generates embeddings for search
- Updates FAISS index incrementally
- No index rebuild required
"""
            )
        return

    st.divider()
    st.markdown(f"### Selected Files ({len(uploaded_files)})")
    
    for idx, file in enumerate(uploaded_files, 1):
        st.write(f"{idx}. **{file.name}** ({file.size / 1024:.1f} KB)")

    st.divider()

    if st.button("Process & Upload", use_container_width=True, key="upload_button"):
        with st.spinner("Processing files..."):
            try:
                summary = handle_pdf_upload(uploaded_files)
                
                # Display results
                st.divider()
                st.markdown("### Upload Summary")
                
                if summary["success"]:
                    st.success("✅ Upload completed successfully!")
                else:
                    st.warning("⚠️ Upload completed with some errors. See details below.")
                
                # Show uploaded files
                if summary["uploaded_files"]:
                    st.markdown("#### Uploaded Files")
                    for filename in summary["uploaded_files"]:
                        st.write(f"✓ {filename}")
                
                # Show processed files
                if summary["processed_files"]:
                    st.markdown("#### Processing Results")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        render_metric_card(
                            "Files Processed",
                            len(summary["processed_files"])
                        )
                    with col2:
                        total_pages = sum(
                            f["pages"]
                            for f in summary["processed_files"]
                        )
                        render_metric_card(
                            "Total Pages",
                            total_pages
                        )
                    with col3:
                        total_limitations = sum(
                            f["limitations"]
                            for f in summary["processed_files"]
                        )
                        render_metric_card(
                            "Limitations Found",
                            total_limitations
                        )
                    with col4:
                        new_chunks = summary.get(
                            "new_chunks",
                            0
                        )
                        render_metric_card(
                            "New Chunks",
                            new_chunks or 0
                        )
                    
                    st.divider()
                    for file_info in summary["processed_files"]:
                        st.markdown(
                            f"""
**{file_info['name']}**
- Pages: {file_info['pages']}
- Limitations: {file_info['limitations']}
- Chunks: {file_info['chunks']}
"""
                        )
                
                # Show skipped files
                if summary["skipped_files"]:
                    st.markdown("#### Skipped Files")
                    st.info("These files were already processed:")
                    for skip_info in summary["skipped_files"]:
                        st.write(f"⊘ {skip_info['name']} - {skip_info['reason']}")
                
                # Show errors
                if summary["errors"]:
                    st.markdown("#### Errors")
                    for error in summary["errors"]:
                        st.error(
                            f"❌ {error['file']}: {error['error']}"
                        )
                
                # Show index update status
                if summary["index_updated"]:
                    st.markdown("#### Index Update")
                    st.success(
                        f"✓ FAISS index updated with "
                        f"{summary['new_chunks']} new chunks"
                    )
                
                # Next steps
                if summary["success"] and summary["processed_files"]:
                    st.divider()
                    st.markdown("### Next Steps")
                    st.markdown(
                        """
1. **View new papers** in the "Limitations" page
2. **Search** using the "Paper Search" page
3. **Update research gaps** by running `src/research_gap_generator.py`
4. **View analytics** on the "Analytics" page
"""
                    )
                    
            except Exception as e:
                st.error(f"❌ Error processing files: {str(e)}")
                st.write("Please check the logs and try again.")


def page_search():
    st.title("Paper Search")
    st.write("Use semantic search to find relevant information across all papers.")
    st.divider()

    col1, col2 = st.columns([4, 1])
    with col1:
        query = st.text_input(
            "Enter your research question",
            placeholder="e.g., What are the main challenges in transformer training?",
            label_visibility="collapsed",
        )
    
    # Advanced search settings in sidebar
    with st.sidebar:
        st.markdown("### Search Settings")
        top_k = st.slider("Number of sources to retrieve", min_value=3, max_value=20, value=10, step=1)
        bm25_weight = st.slider("BM25 Weight (lexical)", min_value=0.0, max_value=1.0, value=0.6, step=0.1)
        faiss_weight = 1.0 - bm25_weight
        st.caption(f"FAISS Weight (semantic): {faiss_weight:.1f}")
    
    with col2:
        search_button = st.button("Search", use_container_width=True)

    if search_button and query:
        with st.spinner("Searching across papers..."):
            retrieved_chunks, answer = perform_semantic_search(
                query, 
                top_k=top_k,
                bm25_weight=bm25_weight,
                faiss_weight=faiss_weight
            )

        if retrieved_chunks:
            st.markdown("### AI-Generated Answer")
            st.info(answer)
            st.divider()
            st.markdown("### Retrieved Sources")

            st.subheader("Sources")
            for chunk in retrieved_chunks:

                st.markdown(
                    f"""
                **{chunk['paper']}**

                Page: {chunk['page']}

                Score: {chunk['score']:.2f}
                """
                )

            st.divider()
            for idx, chunk in enumerate(retrieved_chunks, start=1):
                title = truncate_text(chunk['content'], 80)
                with st.expander(f"Source {idx}: {title}"):
                    st.write(f"**Paper:** {chunk['paper']}")
                    st.write(f"**Page:** {chunk['page']}")
                    st.write("**Content:**")
                    st.write(chunk['content'])
        else:
            st.error(answer)

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
### Search Tips

- Ask focused questions about methods, challenges, or limitations
- Compare models or architectures directly
- Use topic-specific language from your domain
"""
        )
    with col2:
        st.markdown(
            """
### How It Works

1. Your query is embedded semantically
2. FAISS retrieves relevant chunks
3. Gemini generates an answer grounded in retrieved evidence
"""
        )


def page_limitations():
    st.title("Limitations Explorer")
    st.write("Browse and search all research limitations extracted from papers.")
    st.divider()

    search_term = st.text_input(
        "Filter limitations",
        placeholder="Search for specific limitations...",
        label_visibility="collapsed",
    )

    limitations_by_paper = get_limitations_by_paper()
    all_limitations = load_all_limitations()

    if not all_limitations:
        st.warning("No limitations found. Run the extraction pipeline first.")
        return

    st.divider()
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_metric_card("Total Limitations", len(all_limitations))
    with col2:
        render_metric_card("Papers Analyzed", len(limitations_by_paper))
    with col3:
        avg_per_paper = len(all_limitations) // len(limitations_by_paper) if limitations_by_paper else 0
        render_metric_card("Avg per Paper", avg_per_paper)
    with col4:
        max_per_paper = max((len(items) for items in limitations_by_paper.values()), default=0)
        render_metric_card("Max in One Paper", max_per_paper)

    st.divider()
    for paper_name in sorted(limitations_by_paper):
        limitations = limitations_by_paper[paper_name]
        if search_term:
            limitations = [item for item in limitations if search_term.lower() in item.lower()]

        if limitations:
            with st.expander(f"{paper_name} ({len(limitations)} limitations)"):
                for idx, limitation in enumerate(limitations, start=1):
                    st.markdown(f"**Limitation {idx}**")
                    st.write(limitation)
                    if idx != len(limitations):
                        st.divider()


def page_clusters():
    st.title("Research Themes and Clusters")
    st.write("Cluster-level view of the limitation themes stored in the generated output.")
    st.divider()

    clusters = get_cluster_analysis()
    if not clusters:
        st.warning("No cluster output found. Generate `outputs/research_gaps.json` first.")
        return

    total_limitations = sum(len(cluster["limitations"]) for cluster in clusters.values())

    col1, col2, col3 = st.columns(3)
    with col1:
        render_metric_card("Total Themes", len(clusters))
    with col2:
        render_metric_card("Total Limitations", total_limitations)
    with col3:
        avg_cluster_size = total_limitations // len(clusters) if clusters else 0
        render_metric_card("Avg Cluster Size", avg_cluster_size)

    st.divider()
    for cluster_id, cluster_data in sorted(clusters.items()):
        with st.container(border=True):
            st.subheader(cluster_data.get("theme", f"Cluster {cluster_id}"))
            st.write("**Research Gap**")
            st.write(cluster_data.get("research_gap", "Research gap details are unavailable."))
            st.write("**Potential Idea**")
            st.info(cluster_data.get("research_idea", "Potential idea details are unavailable."))
            st.write("**Key Limitations**")
            for limitation in cluster_data.get("limitations", [])[:4]:
                st.write(f"- {truncate_text(limitation, 140)}")
            remaining = len(cluster_data.get("limitations", [])) - 4
            if remaining > 0:
                st.caption(f"... and {remaining} more supporting limitations")


def page_gaps():
    st.title("Research Gaps and Opportunities")
    st.write("Structured research gaps loaded directly from `outputs/research_gaps.json`.")
    st.divider()

    research_gaps = load_research_gaps()
    if not research_gaps:
        st.warning("No research gaps found. Run `src/research_gap_generator.py` first.")
        return

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_metric_card("Total Research Gaps", len(research_gaps))
    with col2:
        limitation_total = sum(len(item["limitations"]) for item in research_gaps)
        render_metric_card("Supporting Limitations", limitation_total)
    with col3:
        avg_cluster_size = limitation_total // len(research_gaps) if research_gaps else 0
        render_metric_card("Avg Evidence per Gap", avg_cluster_size)
    with col4:
        longest_gap = max((len(item["research_gap"].split()) for item in research_gaps), default=0)
        render_metric_card("Longest Gap Summary", longest_gap, "words")

    st.divider()
    st.markdown(f"## {len(research_gaps)} Research Opportunities Identified")

    for gap in sorted(research_gaps, key=lambda item: item["rank"]):
        title = (
            f"🏆 Rank #{gap['rank']} | "
            f"{gap['theme']}"
        )

        with st.expander(
            title,
            expanded=(gap["rank"] == 1)
        ):

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Evidence Score",
                    f"{gap['evidence_score']:.2f}"
                )

            with col2:
                st.metric(
                    "Cluster Size",
                    gap["cluster_size"]
                )

            with col3:
                st.metric(
                    "Source Papers",
                    gap["paper_count"]
                )

            st.divider()

            st.markdown(
                "### Research Gap"
            )

            st.write(
                gap["research_gap"]
            )

            st.markdown(
                "### Proposed Research Direction"
            )

            st.info(
                gap["potential_idea"]
            )

            st.markdown(
                "### Source Papers"
            )

            for paper in gap[
                "supporting_papers"
            ]:
                st.write(
                    f"• {paper}"
                )

            st.markdown(
                "### Supporting Limitations"
            )

            for limitation in gap[
                "limitations"
            ]:
                st.write(
                    f"- {limitation}"
                )

    st.divider()
    st.markdown("## Export Results")

    json_payload = json.dumps(research_gaps, indent=4, ensure_ascii=False)
    csv_payload = get_research_gaps_csv()
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="Download Research Gaps JSON",
            data=json_payload,
            file_name="research_gaps.json",
            mime="application/json",
            use_container_width=True,
        )
    with col2:
        st.download_button(
            label="Download Research Gaps CSV",
            data=csv_payload,
            file_name="research_gaps.csv",
            mime="text/csv",
            use_container_width=True,
        )


def page_analytics():
    st.title("Analytics and Insights")
    st.write("Visual analytics of the extracted limitations and generated research gaps.")
    st.divider()

    analytics_data = get_analytics_data()
    total_papers = len(analytics_data["lim_per_paper"])
    total_limitations = sum(analytics_data["lim_per_paper"].values())
    total_clusters = analytics_data["total_clusters"]
    total_gaps = analytics_data["total_research_gaps"]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_metric_card("Total Papers", total_papers)
    with col2:
        render_metric_card("Total Limitations", total_limitations)
    with col3:
        render_metric_card("Total Clusters", total_clusters)
    with col4:
        render_metric_card("Total Research Gaps", total_gaps)

    if not analytics_data["lim_per_paper"] and not analytics_data["cluster_sizes"]:
        st.warning("Analytics are unavailable until paper outputs and research gap outputs are generated.")
        return

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Limitations per Paper")
        lim_data = analytics_data["lim_per_paper"]
        if lim_data:
            fig = go.Figure(
                data=[
                    go.Bar(
                        x=list(lim_data.keys()),
                        y=list(lim_data.values()),
                        marker=dict(color=list(lim_data.values()), colorscale="Blues"),
                        text=list(lim_data.values()),
                        textposition="auto",
                    )
                ]
            )
            fig.update_layout(
                xaxis_title="Paper",
                yaxis_title="Number of Limitations",
                template="plotly_white",
                height=400,
                showlegend=False,
            )
            fig.update_xaxes(tickangle=-35)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No per-paper limitation data available.")

    with col2:
        st.markdown("### Evidence Distribution by Cluster")
        cluster_sizes = analytics_data["cluster_sizes"]
        if cluster_sizes:
            fig = go.Figure(
                data=[
                    go.Pie(
                        labels=list(cluster_sizes.keys()),
                        values=list(cluster_sizes.values()),
                        hole=0.45,
                        marker=dict(colors=px.colors.sequential.Tealgrn),
                    )
                ]
            )
            fig.update_layout(template="plotly_white", height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No cluster distribution data available.")

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Theme Coverage")
        theme_labels = analytics_data["theme_labels"]
        if theme_labels:
            fig = go.Figure(
                data=[
                    go.Bar(
                        y=list(theme_labels.keys()),
                        x=list(theme_labels.values()),
                        orientation="h",
                        marker=dict(color="#2a7f62"),
                        text=list(theme_labels.values()),
                        textposition="auto",
                    )
                ]
            )
            fig.update_layout(
                xaxis_title="Supporting Limitations",
                yaxis_title="Theme",
                template="plotly_white",
                height=420,
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No theme coverage data available.")

    with col2:
        st.markdown("### Gap Summary Statistics")
        avg_limitations = total_limitations // total_papers if total_papers else 0
        max_limitations = max(analytics_data["lim_per_paper"].values(), default=0)
        avg_gap_words = (
            sum(item["research_gap_length"] for item in analytics_data["gap_records"]) // total_gaps
            if total_gaps
            else 0
        )
        summary_stats = {
            "Average limitations per paper": avg_limitations,
            "Maximum limitations in a paper": max_limitations,
            "Average research gap length (words)": avg_gap_words,
            "Clusters with generated ideas": total_gaps,
        }
        for label, value in summary_stats.items():
            st.write(f"**{label}:** {value}")


def page_reports():
    st.title("📄 Reports")
    st.write("Generate and download research gap reports in PDF and Markdown formats.")
    st.divider()

    research_gaps = load_research_gaps()
    if not research_gaps:
        st.warning("No research gaps found. Run `src/research_gap_generator.py` first.")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### PDF Report")
        st.write("Generate a comprehensive PDF report with executive summary and ranked research gaps.")
        
        if st.button("Generate PDF Report", use_container_width=True, key="generate_pdf"):
            import time
            start_time = time.time()
            with st.spinner("Generating PDF report..."):
                pdf_path = generate_pdf_report()
                generation_time = time.time() - start_time
                if pdf_path:
                    st.success(f"PDF report generated successfully!")
                    st.info(f"File path: {pdf_path}")
                    st.caption(f"Generation time: {generation_time:.2f} seconds")
                else:
                    st.error("Failed to generate PDF report.")

        pdf_path = Path(__file__).resolve().parent.parent / "outputs" / "Research_Gap_Report.pdf"
        if pdf_path.exists():
            with open(pdf_path, "rb") as pdf_file:
                st.download_button(
                    label="Download PDF Report",
                    data=pdf_file,
                    file_name="Research_Gap_Report.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )

    with col2:
        st.markdown("### Literature Review")
        st.write("Generate a structured literature review in Markdown format.")
        
        if st.button("Generate Literature Review", use_container_width=True, key="generate_lit"):
            import time
            start_time = time.time()
            with st.spinner("Generating literature review..."):
                lit_path = generate_literature_review()
                generation_time = time.time() - start_time
                if lit_path:
                    st.success(f"Literature review generated successfully!")
                    st.info(f"File path: {lit_path}")
                    st.caption(f"Generation time: {generation_time:.2f} seconds")
                else:
                    st.error("Failed to generate literature review.")

        lit_path = Path(__file__).resolve().parent.parent / "outputs" / "literature_review.md"
        if lit_path.exists():
            with open(lit_path, "r", encoding="utf-8") as lit_file:
                st.download_button(
                    label="Download Literature Review",
                    data=lit_file.read(),
                    file_name="literature_review.md",
                    mime="text/markdown",
                    use_container_width=True,
                )

    st.divider()
    st.markdown("### Report Information")
    col1, col2, col3 = st.columns(3)
    with col1:
        render_metric_card("Research Gaps", len(research_gaps))
    with col2:
        total_papers = len(set().union(*[gap.get("supporting_papers", []) for gap in research_gaps]))
        render_metric_card("Source Papers", total_papers)
    with col3:
        avg_evidence = sum(gap.get("evidence_score", 0) for gap in research_gaps) / len(research_gaps) if research_gaps else 0
        render_metric_card("Avg Evidence Score", f"{avg_evidence:.2f}")


def main():
    if page == "Home":
        page_home()
    elif page == "Upload PDF":
        page_upload()
    elif page == "Paper Search":
        page_search()
    elif page == "Limitations":
        page_limitations()
    elif page == "Clusters":
        page_clusters()
    elif page == "Research Gaps":
        page_gaps()
    elif page == "Analytics":
        page_analytics()
    elif page == "Reports":
        page_reports()


if __name__ == "__main__":
    main()
