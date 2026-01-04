"""
Extended admin dashboard with raw user data viewer
"""
import streamlit as st
import json
from pathlib import Path
from app_components.user_registration import UserRegistration
import pandas as pd


def render_raw_data_viewer():
    """Render raw user data for admin viewing"""

    st.markdown("### 🔍 Raw User Data")

    storage_file = Path("user_data.json")

    if not storage_file.exists():
        st.warning("No user data found yet.")
        return

    try:
        with open(storage_file, 'r') as f:
            data = json.load(f)

        users = data.get('users', {})

        if not users:
            st.info("No users registered yet.")
            return

        # Convert to DataFrame for better display
        users_list = []
        for user_id, user_data in users.items():
            users_list.append({
                'User ID': user_id,
                'Name': user_data.get('name', 'N/A'),
                'Institution': user_data.get('institution', 'N/A'),
                'Country': user_data.get('country', 'N/A'),
                'Purpose': user_data.get('purpose', 'N/A'),
                'Registered': user_data.get('registered_at', 'N/A'),
                'Skipped': '✓' if user_data.get('skipped', False) else '✗'
            })

        df = pd.DataFrame(users_list)

        # Display count
        st.info(f"Total users: {len(users)}")

        # Search/Filter
        st.markdown("#### 🔎 Search & Filter")
        col1, col2 = st.columns(2)

        with col1:
            search_term = st.text_input("Search by name or institution", "")

        with col2:
            country_filter = st.selectbox(
                "Filter by country",
                ["All"] + sorted(df['Country'].unique().tolist())
            )

        # Apply filters
        filtered_df = df.copy()

        if search_term:
            mask = (
                filtered_df['Name'].str.contains(search_term, case=False, na=False) |
                filtered_df['Institution'].str.contains(search_term, case=False, na=False)
            )
            filtered_df = filtered_df[mask]

        if country_filter != "All":
            filtered_df = filtered_df[filtered_df['Country'] == country_filter]

        # Display table
        st.dataframe(
            filtered_df,
            use_container_width=True,
            hide_index=True,
            height=500
        )

        # Download options
        st.markdown("---")
        st.markdown("#### 💾 Export Options")

        col1, col2, col3 = st.columns(3)

        with col1:
            csv_data = filtered_df.to_csv(index=False)
            st.download_button(
                label="📥 Download as CSV",
                data=csv_data,
                file_name="user_data.csv",
                mime="text/csv",
                use_container_width=True
            )

        with col2:
            json_data = filtered_df.to_json(orient='records', indent=2)
            st.download_button(
                label="📥 Download as JSON",
                data=json_data,
                file_name="user_data.json",
                mime="application/json",
                use_container_width=True
            )

        with col3:
            try:
                excel_buffer = io.BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                    filtered_df.to_excel(writer, index=False, sheet_name='Users')
                excel_data = excel_buffer.getvalue()

                st.download_button(
                    label="📥 Download as Excel",
                    data=excel_data,
                    file_name="user_data.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
            except ImportError:
                st.info("Install openpyxl for Excel export")

    except Exception as e:
        st.error(f"Error loading user data: {str(e)}")


# Add this import at the top
import io
