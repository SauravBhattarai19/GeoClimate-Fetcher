"""
Admin dashboard for viewing user statistics
Access via: ?admin=true
"""
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from app_components.user_registration import UserRegistration


def render_admin_dashboard():
    """Render the admin dashboard with user statistics"""

    st.markdown("""
    <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #0a4d68 0%, #088395 50%, #05bfdb 100%);
         border-radius: 20px; color: white; margin-bottom: 2rem;">
        <h1 style="margin: 0;">📊 Admin Dashboard</h1>
        <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">User Analytics & Statistics</p>
    </div>
    """, unsafe_allow_html=True)

    # Get stats
    user_reg = UserRegistration()
    stats = user_reg.get_stats()

    # Overview metrics
    st.markdown("### 📈 Overview")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Users", stats['total_users'])

    with col2:
        st.metric("Countries", len(stats['countries']))

    with col3:
        st.metric("Use Cases", len(stats['purposes']))

    with col4:
        st.metric("Institutions", len(stats['institutions']))

    st.markdown("---")

    # Geographic Distribution
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🌍 Geographic Distribution")
        if stats['countries']:
            fig = px.bar(
                x=list(stats['countries'].keys()),
                y=list(stats['countries'].values()),
                labels={'x': 'Country', 'y': 'Users'},
                color=list(stats['countries'].values()),
                color_continuous_scale='Teal'
            )
            fig.update_layout(
                showlegend=False,
                height=400,
                xaxis_tickangle=-45
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No country data yet")

    with col2:
        st.markdown("### 🎯 Use Cases")
        if stats['purposes']:
            fig = px.pie(
                values=list(stats['purposes'].values()),
                names=list(stats['purposes'].keys()),
                color_discrete_sequence=px.colors.sequential.Teal
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No purpose data yet")

    # Top Institutions
    st.markdown("### 🏛️ Top Institutions")
    if stats['institutions']:
        # Sort institutions by count
        sorted_inst = sorted(stats['institutions'].items(), key=lambda x: x[1], reverse=True)

        col1, col2 = st.columns([2, 1])

        with col1:
            # Bar chart
            fig = px.bar(
                x=[inst[0] for inst in sorted_inst[:10]],
                y=[inst[1] for inst in sorted_inst[:10]],
                labels={'x': 'Institution', 'y': 'Users'},
                color=[inst[1] for inst in sorted_inst[:10]],
                color_continuous_scale='Teal'
            )
            fig.update_layout(
                showlegend=False,
                height=400,
                xaxis_tickangle=-45
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Table
            st.markdown("**Top 10**")
            for i, (inst, count) in enumerate(sorted_inst[:10], 1):
                st.markdown(f"{i}. **{inst}** ({count} users)")
    else:
        st.info("No institution data yet")

    # Detailed Stats Table
    st.markdown("---")
    st.markdown("### 📋 Detailed Statistics")

    tab1, tab2, tab3, tab4 = st.tabs(["Countries", "Use Cases", "Institutions", "Raw User Data"])

    with tab1:
        if stats['countries']:
            import pandas as pd
            df = pd.DataFrame(
                list(stats['countries'].items()),
                columns=['Country', 'Users']
            ).sort_values('Users', ascending=False)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No data")

    with tab2:
        if stats['purposes']:
            import pandas as pd
            df = pd.DataFrame(
                list(stats['purposes'].items()),
                columns=['Use Case', 'Users']
            ).sort_values('Users', ascending=False)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No data")

    with tab3:
        if stats['institutions']:
            import pandas as pd
            df = pd.DataFrame(
                list(stats['institutions'].items()),
                columns=['Institution', 'Users']
            ).sort_values('Users', ascending=False)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No data")

    with tab4:
        from app_components.raw_data_viewer import render_raw_data_viewer
        render_raw_data_viewer()

    # Export data
    st.markdown("---")
    st.markdown("### 💾 Export Data")

    if st.button("📥 Download Full Report (JSON)", use_container_width=True):
        import json
        json_str = json.dumps(stats, indent=2)
        st.download_button(
            label="Download JSON",
            data=json_str,
            file_name="user_stats.json",
            mime="application/json"
        )

    # Back to platform
    st.markdown("---")
    if st.button("🔙 Back to Platform"):
        # Clear admin query param
        st.query_params.clear()
        st.rerun()
