"""
Streamlit Dashboard - Machine Learning based Buyer Segmentation and
Investment Profiling for Real Estate Market Intelligence
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os

st.set_page_config(page_title="Parcl Buyer Intelligence", page_icon="🏢", layout="wide")

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "clients_segmented.csv")

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

df = load_data()

SEGMENT_COLORS = {
    "Luxury Investors": "#E8A33D",
    "Global Investors": "#2E5EAA",
    "Corporate Buyers": "#3D9970",
    "First-Time / Growth Buyers": "#C4463A",
}

# ---------------------------------------------------------------------------
# SIDEBAR FILTERS
# ---------------------------------------------------------------------------
st.sidebar.title("🏢 Parcl Buyer Intelligence")
st.sidebar.markdown("Filter the buyer universe:")

countries = sorted(df["country"].unique())
regions_all = sorted(df["region"].unique())
purposes = sorted(df["acquisition_purpose"].unique())
client_types = sorted(df["client_type"].unique())
segments = sorted(df["segment_name"].unique())

sel_country = st.sidebar.multiselect("Country", countries, default=[])
available_regions = sorted(df[df["country"].isin(sel_country)]["region"].unique()) if sel_country else regions_all
sel_region = st.sidebar.multiselect("Region", available_regions, default=[])
sel_purpose = st.sidebar.multiselect("Acquisition Purpose", purposes, default=[])
sel_clienttype = st.sidebar.multiselect("Client Type", client_types, default=[])
sel_segment = st.sidebar.multiselect("Buyer Segment", segments, default=[])

filtered = df.copy()
if sel_country:
    filtered = filtered[filtered["country"].isin(sel_country)]
if sel_region:
    filtered = filtered[filtered["region"].isin(sel_region)]
if sel_purpose:
    filtered = filtered[filtered["acquisition_purpose"].isin(sel_purpose)]
if sel_clienttype:
    filtered = filtered[filtered["client_type"].isin(sel_clienttype)]
if sel_segment:
    filtered = filtered[filtered["segment_name"].isin(sel_segment)]

st.sidebar.markdown("---")
st.sidebar.metric("Clients in view", f"{len(filtered):,}")
st.sidebar.caption(f"of {len(df):,} total clients")

# ---------------------------------------------------------------------------
# HEADER + KPIs
# ---------------------------------------------------------------------------
st.title("Machine Learning based Buyer Segmentation & Investment Profiling")
st.caption("Real Estate Market Intelligence · Parcl Co. Limited")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Clients", f"{len(filtered):,}")
k2.metric("Total Investment Value", f"${filtered['total_investment_value'].sum()/1e6:,.1f}M")
k3.metric("Avg. Unit Price", f"${filtered['avg_unit_price'].mean():,.0f}" if len(filtered) else "$0")
k4.metric("Avg. Satisfaction", f"{filtered['satisfaction_score'].mean():.2f} / 5" if len(filtered) else "N/A")

st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Buyer Segmentation Overview",
    "💰 Investor Behavior Dashboard",
    "🌍 Geographic Buyer Analysis",
    "🔎 Segment Insights Panel",
])

# ---------------------------------------------------------------------------
# TAB 1: BUYER SEGMENTATION OVERVIEW
# ---------------------------------------------------------------------------
with tab1:
    st.subheader("Cluster Distribution")
    if len(filtered) == 0:
        st.warning("No clients match the selected filters.")
    else:
        c1, c2 = st.columns([1, 1])
        with c1:
            seg_counts = filtered["segment_name"].value_counts().reset_index()
            seg_counts.columns = ["segment_name", "count"]
            fig = px.pie(seg_counts, names="segment_name", values="count", hole=0.45,
                         color="segment_name", color_discrete_map=SEGMENT_COLORS,
                         title="Share of Clients by Segment")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig2 = px.bar(seg_counts.sort_values("count"), x="count", y="segment_name",
                          orientation="h", color="segment_name", color_discrete_map=SEGMENT_COLORS,
                          title="Client Count by Segment", labels={"count": "Clients", "segment_name": ""})
            fig2.update_layout(showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Segment Characteristics at a Glance")
        summary = filtered.groupby("segment_name").agg(
            clients=("client_id", "count"),
            avg_age=("age", "mean"),
            pct_investment_purpose=("acquisition_purpose", lambda s: (s == "Investment").mean() * 100),
            pct_loan=("loan_flag", "mean"),
            avg_satisfaction=("satisfaction_score", "mean"),
            avg_investment_value=("total_investment_value", "mean"),
        ).round(2)
        summary["pct_loan"] = (summary["pct_loan"] * 100).round(1)
        summary = summary.rename(columns={
            "clients": "Clients", "avg_age": "Avg Age", "pct_investment_purpose": "% Investment Purpose",
            "pct_loan": "% Loan Financed", "avg_satisfaction": "Avg Satisfaction",
            "avg_investment_value": "Avg Investment Value ($)"
        })
        st.dataframe(summary.style.format({
            "Avg Age": "{:.1f}", "% Investment Purpose": "{:.1f}%", "% Loan Financed": "{:.1f}%",
            "Avg Satisfaction": "{:.2f}", "Avg Investment Value ($)": "${:,.0f}"
        }), use_container_width=True)

        st.subheader("Age Distribution by Segment")
        fig3 = px.histogram(filtered, x="age", color="segment_name", nbins=25, barmode="overlay",
                             color_discrete_map=SEGMENT_COLORS, opacity=0.7)
        st.plotly_chart(fig3, use_container_width=True)

# ---------------------------------------------------------------------------
# TAB 2: INVESTOR BEHAVIOR DASHBOARD
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("Investment Patterns by Cluster")
    if len(filtered) == 0:
        st.warning("No clients match the selected filters.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            fig4 = px.box(filtered, x="segment_name", y="total_investment_value", color="segment_name",
                          color_discrete_map=SEGMENT_COLORS, title="Total Investment Value by Segment",
                          labels={"total_investment_value": "Total Investment Value ($)", "segment_name": ""})
            fig4.update_layout(showlegend=False)
            st.plotly_chart(fig4, use_container_width=True)
        with c2:
            fig5 = px.box(filtered, x="segment_name", y="avg_unit_price", color="segment_name",
                          color_discrete_map=SEGMENT_COLORS, title="Average Unit Price by Segment",
                          labels={"avg_unit_price": "Avg Unit Price ($)", "segment_name": ""})
            fig5.update_layout(showlegend=False)
            st.plotly_chart(fig5, use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            loan_ct = pd.crosstab(filtered["segment_name"], filtered["loan_applied"], normalize="index") * 100
            loan_ct = loan_ct.reset_index().melt(id_vars="segment_name", var_name="Loan Applied", value_name="pct")
            fig6 = px.bar(loan_ct, x="segment_name", y="pct", color="Loan Applied", barmode="stack",
                          title="Loan Financing Mix by Segment", labels={"pct": "% of Clients", "segment_name": ""})
            st.plotly_chart(fig6, use_container_width=True)
        with c4:
            purpose_ct = pd.crosstab(filtered["segment_name"], filtered["acquisition_purpose"], normalize="index") * 100
            purpose_ct = purpose_ct.reset_index().melt(id_vars="segment_name", var_name="Purpose", value_name="pct")
            fig7 = px.bar(purpose_ct, x="segment_name", y="pct", color="Purpose", barmode="stack",
                          title="Acquisition Purpose Mix by Segment", labels={"pct": "% of Clients", "segment_name": ""})
            st.plotly_chart(fig7, use_container_width=True)

        st.subheader("Referral Channel Mix by Segment")
        ref_ct = pd.crosstab(filtered["segment_name"], filtered["referral_channel"], normalize="index") * 100
        ref_ct = ref_ct.reset_index().melt(id_vars="segment_name", var_name="Channel", value_name="pct")
        fig8 = px.bar(ref_ct, x="segment_name", y="pct", color="Channel", barmode="stack",
                      labels={"pct": "% of Clients", "segment_name": ""})
        st.plotly_chart(fig8, use_container_width=True)

        st.subheader("Units Purchased vs Investment Value")
        fig9 = px.scatter(filtered, x="total_units_purchased", y="total_investment_value",
                          color="segment_name", color_discrete_map=SEGMENT_COLORS,
                          hover_data=["client_id", "country", "region"],
                          labels={"total_units_purchased": "Units Purchased",
                                  "total_investment_value": "Total Investment Value ($)"})
        st.plotly_chart(fig9, use_container_width=True)

# ---------------------------------------------------------------------------
# TAB 3: GEOGRAPHIC BUYER ANALYSIS
# ---------------------------------------------------------------------------
with tab3:
    st.subheader("Buyer Segments by Geography")
    if len(filtered) == 0:
        st.warning("No clients match the selected filters.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            country_counts = filtered["country"].value_counts().reset_index()
            country_counts.columns = ["country", "count"]
            fig10 = px.choropleth(country_counts, locations="country", locationmode="country names",
                                  color="count", color_continuous_scale="Blues",
                                  title="Client Density by Country")
            st.plotly_chart(fig10, use_container_width=True)
        with c2:
            geo_seg = filtered.groupby(["country", "segment_name"]).size().reset_index(name="count")
            fig11 = px.bar(geo_seg, x="country", y="count", color="segment_name",
                           color_discrete_map=SEGMENT_COLORS, title="Segment Mix by Country",
                           labels={"count": "Clients", "country": ""})
            st.plotly_chart(fig11, use_container_width=True)

        st.subheader("Investment Value by Country")
        inv_by_country = filtered.groupby("country")["total_investment_value"].sum().reset_index()
        inv_by_country = inv_by_country.sort_values("total_investment_value", ascending=False)
        fig12 = px.bar(inv_by_country, x="country", y="total_investment_value",
                       labels={"total_investment_value": "Total Investment Value ($)", "country": ""},
                       color_discrete_sequence=["#2E5EAA"])
        st.plotly_chart(fig12, use_container_width=True)

        st.subheader("Top Regions by Client Count")
        top_regions = filtered["region"].value_counts().head(15).reset_index()
        top_regions.columns = ["region", "count"]
        fig13 = px.bar(top_regions.sort_values("count"), x="count", y="region", orientation="h",
                       color_discrete_sequence=["#3D9970"], labels={"count": "Clients", "region": ""})
        st.plotly_chart(fig13, use_container_width=True)

# ---------------------------------------------------------------------------
# TAB 4: SEGMENT INSIGHTS PANEL
# ---------------------------------------------------------------------------
with tab4:
    st.subheader("Descriptive Statistics per Segment")
    if len(filtered) == 0:
        st.warning("No clients match the selected filters.")
    else:
        chosen_segment = st.selectbox("Select a segment to inspect", sorted(filtered["segment_name"].unique()))
        seg_df = filtered[filtered["segment_name"] == chosen_segment]

        SEGMENT_DESCRIPTIONS = {
            "Luxury Investors": "High-value, high-satisfaction buyers with the largest average deal sizes. Primarily website/direct sourced, low loan dependency, strong candidates for premium/off-market listings and concierge service.",
            "Global Investors": "Agency-sourced, cross-border oriented buyers with moderate-to-high investment share. Best reached through broker/agency partnerships and relocation-linked marketing.",
            "Corporate Buyers": "Company-registered clients purchasing multiple units, often for portfolio or leasing purposes. Prioritize bulk-deal terms, dedicated account management, and B2B contracts.",
            "First-Time / Growth Buyers": "The largest segment; more loan-dependent, lower average price point, higher share of investment-purpose purchases. Strong fit for financing partnerships, entry-level inventory, and digital-first marketing.",
        }
        st.info(SEGMENT_DESCRIPTIONS.get(chosen_segment, ""))

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Clients", f"{len(seg_df):,}")
        c2.metric("Avg Age", f"{seg_df['age'].mean():.1f}")
        c3.metric("Avg Investment Value", f"${seg_df['total_investment_value'].mean():,.0f}")
        c4.metric("Avg Satisfaction", f"{seg_df['satisfaction_score'].mean():.2f} / 5")

        st.markdown("#### Full Descriptive Statistics")
        num_cols = ["age", "satisfaction_score", "total_units_purchased", "total_investment_value",
                    "avg_unit_price", "avg_floor_area"]
        st.dataframe(seg_df[num_cols].describe().T.round(2), use_container_width=True)

        c5, c6 = st.columns(2)
        with c5:
            st.markdown("#### Client Type Mix")
            st.bar_chart(seg_df["client_type"].value_counts())
        with c6:
            st.markdown("#### Gender Mix")
            st.bar_chart(seg_df["gender"].value_counts())

        st.markdown("#### Raw Client Records")
        display_cols = ["client_id", "client_type", "gender", "age", "country", "region",
                         "acquisition_purpose", "loan_applied", "referral_channel",
                         "satisfaction_score", "total_units_purchased", "total_investment_value",
                         "avg_unit_price", "segment_name"]
        st.dataframe(seg_df[display_cols], use_container_width=True, height=350)

        csv = seg_df[display_cols].to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download this segment as CSV", csv,
                          file_name=f"{chosen_segment.replace(' ', '_').replace('/','')}_clients.csv",
                          mime="text/csv")

st.markdown("---")
st.caption("Machine Learning based Buyer Segmentation and Investment Profiling · "
           "Unified Mentor × Parcl Co. Limited")
