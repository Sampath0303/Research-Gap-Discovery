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
        ["Home", "Paper Search", "Limitations", "Clusters", "Research Gaps", "Analytics"],
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
    with col2:
        search_button = st.button("Search", use_container_width=True)

    if search_button and query:
        with st.spinner("Searching across papers..."):
            retrieved_chunks, answer = perform_semantic_search(query, top_k=5)

        if retrieved_chunks:
            st.markdown("### AI-Generated Answer")
            st.info(answer)
            st.divider()
            st.markdown("### Retrieved Sources")

            for idx, chunk in enumerate(retrieved_chunks, start=1):
                title = truncate_text(chunk.page_content, 80)
                with st.expander(f"Source {idx}: {title}"):
                    st.write(f"**Paper:** {chunk.metadata.get('source_file', 'Unknown')}")
                    st.write("**Content:**")
                    st.write(chunk.page_content)
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

    for gap in sorted(research_gaps, key=lambda item: item["cluster"]):
        title = f"Cluster {gap['cluster']}: {gap['theme']}"
        with st.expander(title, expanded=False):
            st.markdown("**Theme**")
            st.write(gap["theme"])
            st.markdown("**Research Gap**")
            st.write(gap["research_gap"])
            st.markdown("**Potential Idea**")
            st.info(gap["potential_idea"])
            st.markdown("**Supporting Limitations**")
            for limitation in gap["limitations"]:
                st.write(f"- {format_limitation(limitation, max_lines=2)}")

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


def main():
    if page == "Home":
        page_home()
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


if __name__ == "__main__":
    main()
