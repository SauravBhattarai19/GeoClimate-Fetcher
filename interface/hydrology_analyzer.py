"""
Hydrology Analyzer Interface Module
Handles the complete interface for the Hydrology Analyzer tool
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, date, timedelta
import ee

# Import core components
from geoclimate_fetcher.core import GeometrySelectionWidget
from geoclimate_fetcher.hydrology_analysis import HydrologyAnalyzer

# Import post-download integration
from app_components.post_download_integration import (
    get_download_handler,
    register_csv_download,
    render_post_download_integration
)
from app_components.quick_visualization import quick_visualizer

# Import smart download components
from app_components.download_component import DownloadHelper


def render_hydrology_analyzer():
    """Render the complete Hydrology Analyzer interface"""

    # App title and header
    st.markdown('<h1 class="main-title">💧 Hydrology Analyzer</h1>', unsafe_allow_html=True)
    st.markdown("### Comprehensive precipitation analysis for hydrology education and research")
    
    st.markdown("---")
    
    # Configuration Section
    with st.container():
        st.markdown("## 🔧 Analysis Configuration")

        # Step 1: Area selection (full width for map visibility)
        geometry_ready = _render_hydrology_geometry_selection()

        # Only show dataset and date selection after geometry is selected
        if geometry_ready:
            st.markdown("---")
            # Two columns for dataset and dates
            col2, col3 = st.columns([1, 1])

            with col2:
                dataset_ready = _render_hydrology_dataset_selection()

            with col3:
                dates_ready = _render_hydrology_date_selection()
        else:
            dataset_ready = False
            dates_ready = False
    
    # Analysis readiness check
    analysis_ready = geometry_ready and dataset_ready and dates_ready
    
    st.markdown("---")
    
    # Analysis controls
    if analysis_ready:
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            if st.button("🚀 Run Hydrology Analysis", type="primary", use_container_width=True):
                # Trigger analysis
                st.session_state.hydro_run_analysis = True
                st.rerun()
        
        with col2:
            if st.button("🔄 Reset Configuration", use_container_width=True):
                _reset_hydrology_configuration()
                st.rerun()
        
        with col3:
            if st.button("📊 View Sample Results", use_container_width=True):
                st.session_state.hydro_show_sample = True
                st.rerun()
    else:
        st.info("👆 Please complete the configuration above to enable analysis")
    
    st.markdown("---")
    
    # Results section - show after analysis is ready or sample requested
    if analysis_ready:
        if st.session_state.get('hydro_run_analysis', False) or st.session_state.get('hydro_analysis_ready', False):
            _render_hydrology_results()
        elif st.session_state.get('hydro_show_sample', False):
            _render_sample_results()
        else:
            # Show configuration summary without fetching data
            st.markdown("## 📋 Ready for Analysis")
            st.info("👆 Click **'🚀 Run Hydrology Analysis'** above to start downloading data and performing the analysis.")

            # Show what will be analyzed
            with st.expander("📊 Analysis Preview", expanded=True):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown("**Dataset:**")
                    dataset_info = st.session_state.hydro_dataset_info
                    st.info(f"{dataset_info['name']}\n({dataset_info['provider']})")
                with col2:
                    st.markdown("**Analysis Period:**")
                    years = (st.session_state.hydro_end_date - st.session_state.hydro_start_date).days / 365.25
                    st.info(f"{st.session_state.hydro_start_date} to\n{st.session_state.hydro_end_date}\n({years:.1f} years)")
                with col3:
                    st.markdown("**Study Area:**")
                    try:
                        area_km2 = st.session_state.hydro_geometry.area().divide(1000000).getInfo()
                        st.info(f"{area_km2:.2f} km²")
                    except:
                        st.info("Selected area")


def _render_hydrology_geometry_selection():
    """Render geometry selection for hydrology analysis"""
    # Check if we already have a geometry
    if hasattr(st.session_state, 'hydro_geometry') and st.session_state.hydro_geometry is not None:
        geometry_ready = True
        st.success("✅ Area ready for analysis")
        try:
            area_km2 = st.session_state.hydro_geometry.area().divide(1000000).getInfo()
            st.metric("Current Area", f"{area_km2:.2f} km²")
        except:
            st.info("Geometry ready for analysis")

        # Add reset button
        if st.button("🗑️ Reset Area", key="hydro_reset_geometry"):
            del st.session_state.hydro_geometry
            if 'hydro_area_km2' in st.session_state:
                del st.session_state.hydro_area_km2
            geometry_ready = False
            st.rerun()
    else:
        def on_geometry_selected(geometry):
            """Callback when geometry is selected"""
            st.session_state.hydro_geometry = geometry
            try:
                area_km2 = geometry.area().divide(1000000).getInfo()
                st.session_state.hydro_area_km2 = area_km2
                st.success(f"✅ Area selected: {area_km2:.2f} km²")
            except:
                st.success("✅ Area selected successfully!")

        # Use the unified geometry selection widget
        geometry_widget = GeometrySelectionWidget(
            session_prefix="hydro_",
            title="🗺️ Area of Interest"
        )

        if geometry_widget.render_complete_interface(on_geometry_selected=on_geometry_selected):
            st.rerun()
            geometry_ready = True
        else:
            geometry_ready = False

    return geometry_ready


def _render_hydrology_dataset_selection():
    """Render dataset selection for hydrology analysis"""
    st.markdown("### 📊 Precipitation Dataset")
    
    # Load precipitation datasets
    precipitation_datasets = _get_precipitation_datasets()
    
    if precipitation_datasets:
        # Dataset selection
        dataset_names = [d['name'] for d in precipitation_datasets]
        selected_name = st.selectbox(
            "Choose dataset:",
            dataset_names,
            key="hydro_dataset_selector"
        )
        
        # Find selected dataset info
        selected_dataset = next(d for d in precipitation_datasets if d['name'] == selected_name)
        
        # Show dataset info
        with st.expander("📋 Dataset Details", expanded=False):
            st.markdown(f"**Provider:** {selected_dataset['provider']}")
            st.markdown(f"**Resolution:** {selected_dataset['resolution']}")
            st.markdown(f"**Period:** {selected_dataset['period']}")
            st.markdown(f"**Description:** {selected_dataset['description']}")
        
        # Check if dataset changed and clear cached data if so
        previous_dataset = st.session_state.get('hydro_dataset_info', {})
        if previous_dataset.get('ee_id') != selected_dataset['ee_id']:
            # Dataset changed, clear cached data
            if 'hydro_precipitation_data' in st.session_state:
                del st.session_state.hydro_precipitation_data
            if 'hydro_analyzer' in st.session_state:
                del st.session_state.hydro_analyzer

        # Store selected dataset
        st.session_state.hydro_dataset_info = selected_dataset

        st.success(f"✅ Dataset: {selected_name}")
        return True
    else:
        st.error("❌ No precipitation datasets available")
        return False


def _render_hydrology_date_selection():
    """Render date selection for hydrology analysis"""
    st.markdown("### 📅 Analysis Period")

    # Get selected dataset info to determine date constraints
    dataset_info = st.session_state.get('hydro_dataset_info', {})

    # Parse dataset date constraints
    from datetime import datetime
    if 'start_date' in dataset_info:
        dataset_min_date = datetime.strptime(dataset_info['start_date'], '%Y-%m-%d').date()
        dataset_max_date = datetime.strptime(dataset_info['end_date'], '%Y-%m-%d').date()
    else:
        # Fallback to most restrictive range
        dataset_min_date = date(2000, 6, 1)  # IMERG start
        dataset_max_date = date.today()

    # Show dataset date availability
    st.info(f"📊 **{dataset_info.get('name', 'Selected dataset')}** is available from **{dataset_min_date}** to **{dataset_max_date}**")

    # Quick date options
    date_option = st.radio(
        "Select period:",
        ["Last 5 years", "Last 10 years", "Custom"],
        key="hydro_date_option",
        horizontal=True
    )

    today = date.today()

    if date_option == "Last 5 years":
        start_date = max(dataset_min_date, date(today.year - 5, 1, 1))
        end_date = min(dataset_max_date, date(today.year - 1, 12, 31))
    elif date_option == "Last 10 years":
        start_date = max(dataset_min_date, date(today.year - 10, 1, 1))
        end_date = min(dataset_max_date, date(today.year - 1, 12, 31))
    else:  # Custom
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=st.session_state.get('hydro_start_date', max(dataset_min_date, date(today.year - 5, 1, 1))),
                min_value=dataset_min_date,
                max_value=dataset_max_date,
                key="hydro_start_date_input"
            )
        with col2:
            end_date = st.date_input(
                "End Date",
                value=st.session_state.get('hydro_end_date', min(dataset_max_date, date(today.year - 1, 12, 31))),
                min_value=dataset_min_date,
                max_value=dataset_max_date,
                key="hydro_end_date_input"
            )
    
    # Validate dates
    if start_date >= end_date:
        st.error("❌ Start date must be before end date")
        return False

    # Check if dates are within dataset availability
    if start_date < dataset_min_date or end_date > dataset_max_date:
        st.error(f"❌ Selected dates are outside dataset availability ({dataset_min_date} to {dataset_max_date})")
        return False

    # Check minimum period
    years_diff = (end_date - start_date).days / 365.25
    if years_diff < 2:
        st.warning("⚠️ Minimum 2 years recommended for reliable analysis")

    # Show specific dataset recommendations
    if 'GPM IMERG' in dataset_info.get('name', ''):
        if start_date < date(2000, 6, 1):
            st.warning(f"⚠️ GPM IMERG data starts from June 1, 2000. Adjusting start date.")
            start_date = date(2000, 6, 1)
    
    # Check if dates changed and clear cached data if so
    previous_start = st.session_state.get('hydro_start_date')
    previous_end = st.session_state.get('hydro_end_date')
    if previous_start != start_date or previous_end != end_date:
        # Date range changed, clear cached data
        if 'hydro_precipitation_data' in st.session_state:
            del st.session_state.hydro_precipitation_data
        if 'hydro_analyzer' in st.session_state:
            del st.session_state.hydro_analyzer

    # Store dates
    st.session_state.hydro_start_date = start_date
    st.session_state.hydro_end_date = end_date

    st.success(f"✅ Period: {years_diff:.1f} years")
    return True


def _render_hydrology_results():
    """Render hydrology analysis results"""
    # Check if we need to fetch new data
    need_new_data = (
        'hydro_precipitation_data' not in st.session_state or
        st.session_state.hydro_precipitation_data is None or
        'hydro_analyzer' not in st.session_state or
        st.session_state.hydro_analyzer is None
    )

    if need_new_data:
        with st.spinner("🔄 Fetching precipitation data from Google Earth Engine..."):
            try:
                # Ensure Earth Engine is authenticated
                from geoclimate_fetcher.core import authenticate

                try:
                    authenticate()
                except:
                    st.warning("⚠️ Earth Engine authentication may be required. Proceeding with available credentials...")

                # Initialize analyzer
                analyzer = HydrologyAnalyzer(st.session_state.hydro_geometry)

                # Fetch data
                dataset_info = st.session_state.hydro_dataset_info
                start_date_str = st.session_state.hydro_start_date.strftime('%Y-%m-%d')
                end_date_str = st.session_state.hydro_end_date.strftime('%Y-%m-%d')

                # Use real data fetching from HydrologyAnalyzer
                st.info(f"📊 Downloading {dataset_info['name']} data for the selected area...")
                precipitation_data = analyzer.fetch_precipitation_data(dataset_info, start_date_str, end_date_str)

                if precipitation_data is not None and not precipitation_data.empty:
                    st.session_state.hydro_precipitation_data = precipitation_data
                    st.session_state.hydro_analyzer = analyzer  # Store analyzer for use in analysis
                    st.success(f"✅ Successfully downloaded {len(precipitation_data)} days of precipitation data!")

                    # Show data summary
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Records", len(precipitation_data))
                    with col2:
                        st.metric("Date Range", f"{len(precipitation_data)} days")
                    with col3:
                        avg_precip = precipitation_data['precipitation'].mean()
                        st.metric("Average Precip", f"{avg_precip:.2f} mm")

                    st.success(f"✅ Data download complete! Proceeding to analysis...")

                    # Mark analysis as ready to display results directly
                    st.session_state.hydro_analysis_ready = True

                else:
                    st.error("❌ No precipitation data was retrieved. Please try a different date range or dataset.")
                    return

            except Exception as e:
                st.error(f"❌ Error fetching data: {str(e)}")
                st.info("💡 This might be due to:")
                st.info("- Earth Engine authentication issues")
                st.info("- Network connectivity problems")
                st.info("- Invalid date range for the selected dataset")
                st.info("- Area too large for the time period selected")

                # Offer fallback option
                if st.button("📊 Use Sample Data Instead", key="use_sample_data"):
                    precipitation_data = _simulate_precipitation_data(start_date_str, end_date_str)
                    st.session_state.hydro_precipitation_data = precipitation_data
                    # Create analyzer with simulated data for consistency
                    analyzer = HydrologyAnalyzer(st.session_state.hydro_geometry)
                    analyzer.precipitation_data = precipitation_data
                    st.session_state.hydro_analyzer = analyzer
                    st.warning("⚠️ Using simulated data for demonstration")
                    st.rerun()
                return

    # Display results header with refresh option
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("## 📊 Hydrology Analysis Results")
    with col2:
        if st.button("🔄 Refresh Data", help="Re-download data with current settings"):
            # Clear cached data to force refresh
            if 'hydro_precipitation_data' in st.session_state:
                del st.session_state.hydro_precipitation_data
            if 'hydro_analyzer' in st.session_state:
                del st.session_state.hydro_analyzer
            st.rerun()

    # Show current analysis configuration
    with st.expander("📋 Current Analysis Configuration", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Dataset:**")
            dataset_info = st.session_state.hydro_dataset_info
            st.info(f"{dataset_info['name']}\n({dataset_info['provider']})")
        with col2:
            st.markdown("**Analysis Period:**")
            years = (st.session_state.hydro_end_date - st.session_state.hydro_start_date).days / 365.25
            st.info(f"{st.session_state.hydro_start_date} to\n{st.session_state.hydro_end_date}\n({years:.1f} years)")
        with col3:
            st.markdown("**Study Area:**")
            try:
                area_km2 = st.session_state.hydro_geometry.area().divide(1000000).getInfo()
                st.info(f"{area_km2:.2f} km²")
            except:
                st.info("Selected area")

    # Analysis tabs - ENHANCED with new advanced analyses
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📈 Time Series",
        "📊 Statistics",
        "🌧️ Return Periods",
        "📉 IDF Curves",
        "📈 Trends (Mann-Kendall)",
        "🔄 Change Points (Pettitt)",
        "💧 Flow Duration & Spells",
        "💾 Downloads"
    ])

    with tab1:
        _render_time_series_analysis()

    with tab2:
        _render_statistical_analysis()

    with tab3:
        _render_return_period_analysis()

    with tab4:
        _render_idf_analysis()

    with tab5:
        _render_mann_kendall_analysis()

    with tab6:
        _render_pettitt_analysis()

    with tab7:
        _render_flow_duration_and_spells()

    with tab8:
        _render_hydrology_downloads()


def _render_time_series_analysis():
    """Render time series analysis"""
    st.markdown("### 📈 Precipitation Time Series")

    precipitation_data = st.session_state.hydro_precipitation_data

    # Get available years for filtering
    precipitation_data['date'] = pd.to_datetime(precipitation_data['date'])
    precipitation_data['year'] = precipitation_data['date'].dt.year
    available_years = sorted(precipitation_data['year'].unique())

    # Year filter for daily chart
    year_options = ['All Years'] + [str(year) for year in available_years]
    selected_year = st.selectbox(
        "🗓️ Filter by Year (Daily Chart):",
        options=year_options,
        index=0,  # Default to "All Years"
        key="daily_year_filter"
    )

    # Filter data based on selection
    if selected_year == 'All Years':
        filtered_data = precipitation_data
        chart_title = "Daily Precipitation Time Series (All Years)"
    else:
        filtered_data = precipitation_data[precipitation_data['year'] == int(selected_year)]
        chart_title = f"Daily Precipitation Time Series ({selected_year})"

    # Create time series plot
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=filtered_data['date'],
        y=filtered_data['precipitation'],
        mode='lines',
        name='Daily Precipitation',
        line=dict(color='blue', width=1)
    ))

    fig.update_layout(
        title=chart_title,
        xaxis_title="Date",
        yaxis_title="Precipitation (mm)",
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)

    # Show data summary for filtered view
    if selected_year != 'All Years':
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Days in Year", len(filtered_data))
        with col2:
            st.metric("Total Precipitation", f"{filtered_data['precipitation'].sum():.1f} mm")
        with col3:
            st.metric("Max Daily", f"{filtered_data['precipitation'].max():.1f} mm")
    
    # Monthly aggregation
    st.markdown("### 📅 Monthly Analysis")

    # Year filter for monthly chart
    monthly_year_options = ['All Years'] + [str(year) for year in available_years]
    selected_monthly_year = st.selectbox(
        "🗓️ Filter by Year (Monthly Chart):",
        options=monthly_year_options,
        index=0,  # Default to "All Years"
        key="monthly_year_filter"
    )

    # Filter data for monthly analysis
    if selected_monthly_year == 'All Years':
        monthly_source_data = precipitation_data
        monthly_chart_title = "Monthly Precipitation Totals (All Years)"
    else:
        monthly_source_data = precipitation_data[precipitation_data['year'] == int(selected_monthly_year)]
        monthly_chart_title = f"Monthly Precipitation Totals ({selected_monthly_year})"

    # Calculate monthly stats for filtered data
    monthly_data = _calculate_monthly_stats(monthly_source_data)

    # Update month labels for single year view
    if selected_monthly_year != 'All Years' and len(monthly_data) > 0:
        # For single year, show just month names instead of YYYY-MM
        try:
            # Check if month column contains datetime-parseable strings
            if monthly_data['month'].dtype == 'object':
                # Try to convert to datetime, handling different formats
                monthly_data['month_dt'] = pd.to_datetime(monthly_data['month'], errors='coerce')
                if not monthly_data['month_dt'].isna().all():
                    monthly_data['month_short'] = monthly_data['month_dt'].dt.strftime('%b')
                    x_data = monthly_data['month_short']
                else:
                    # If conversion fails, use original month data
                    x_data = monthly_data['month']
            else:
                # If already datetime, convert directly
                monthly_data['month_short'] = monthly_data['month'].dt.strftime('%b')
                x_data = monthly_data['month_short']
            x_title = "Month"
        except Exception as e:
            # Fallback to original month data if any error occurs
            x_data = monthly_data['month']
            x_title = "Month"
    else:
        x_data = monthly_data['month']
        x_title = "Month"

    fig_monthly = go.Figure()
    fig_monthly.add_trace(go.Bar(
        x=x_data,
        y=monthly_data['total_precip'],
        name='Monthly Total',
        marker_color='lightblue',
        text=monthly_data['total_precip'].round(1),
        textposition='auto'
    ))

    fig_monthly.update_layout(
        title=monthly_chart_title,
        xaxis_title=x_title,
        yaxis_title="Precipitation (mm)",
        height=400
    )

    st.plotly_chart(fig_monthly, use_container_width=True)

    # Show monthly summary for filtered view
    if selected_monthly_year != 'All Years':
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Months with Data", len(monthly_data))
        with col2:
            st.metric("Annual Total", f"{monthly_data['total_precip'].sum():.1f} mm")
        with col3:
            if len(monthly_data) > 0:
                wettest_idx = monthly_data['total_precip'].idxmax()
                try:
                    # Try to use short month name if available
                    if 'month_short' in monthly_data.columns:
                        wettest_month = monthly_data.loc[wettest_idx, 'month_short']
                    else:
                        # Extract month name from original month column
                        month_val = monthly_data.loc[wettest_idx, 'month']
                        if isinstance(month_val, str) and '-' in month_val:
                            # If format is YYYY-MM, extract month number and convert
                            try:
                                month_num = int(month_val.split('-')[1])
                                wettest_month = pd.to_datetime(f"2000-{month_num:02d}-01").strftime('%b')
                            except:
                                wettest_month = month_val
                        else:
                            wettest_month = str(month_val)

                    wettest_amount = monthly_data['total_precip'].max()
                    st.metric("Wettest Month", f"{wettest_month}: {wettest_amount:.1f} mm")
                except Exception as e:
                    # Fallback: just show the amount
                    wettest_amount = monthly_data['total_precip'].max()
                    st.metric("Wettest Month", f"{wettest_amount:.1f} mm")

    # Yearly analysis with trends
    st.markdown("### 📊 Yearly Analysis")

    analyzer = st.session_state.get('hydro_analyzer')
    if analyzer:
        try:
            yearly_stats = analyzer.calculate_yearly_statistics()

            if yearly_stats and 'yearly_data' in yearly_stats and len(yearly_stats['yearly_data']) > 0:
                yearly_data = yearly_stats['yearly_data']
                trends = yearly_stats['trends']

                # Create subplot with multiple metrics
                from plotly.subplots import make_subplots

                fig_yearly = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=('Annual Maximum', 'Annual Mean', 'Annual Total', 'Annual Median'),
                    vertical_spacing=0.1,
                    horizontal_spacing=0.1
                )

                # Color scheme for different metrics
                colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4']

                # Plot each metric with trend line
                metrics = [
                    ('max', 'Annual Maximum (mm)', 1, 1),
                    ('mean', 'Annual Mean (mm/day)', 1, 2),
                    ('total', 'Annual Total (mm)', 2, 1),
                    ('median', 'Annual Median (mm/day)', 2, 2)
                ]

                for i, (metric, ylabel, row, col) in enumerate(metrics):
                    years = yearly_data['year']
                    values = yearly_data[metric]

                    # Add data points
                    fig_yearly.add_scatter(
                        x=years, y=values,
                        mode='markers+lines',
                        name=f'{ylabel}',
                        marker=dict(size=8, color=colors[i]),
                        line=dict(width=2, color=colors[i]),
                        row=row, col=col,
                        showlegend=False
                    )

                    # Add trend line if available
                    if metric in trends and trends[metric]['trend'] != 'insufficient_data':
                        slope = trends[metric]['slope']
                        trend_name = trends[metric]['trend']
                        direction = trends[metric]['direction']

                        # Calculate trend line points
                        y_trend = slope * (years - years.iloc[0]) + values.iloc[0]

                        fig_yearly.add_scatter(
                            x=years, y=y_trend,
                            mode='lines',
                            name=f'Trend ({direction} {trend_name})',
                            line=dict(width=3, dash='dash', color='red'),
                            row=row, col=col,
                            showlegend=False
                        )

                        # Add trend annotation
                        fig_yearly.add_annotation(
                            x=years.iloc[-1], y=y_trend.iloc[-1],
                            text=f"{direction} {trend_name}",
                            showarrow=True,
                            arrowhead=2,
                            arrowsize=1,
                            arrowcolor='red',
                            font=dict(size=10, color='red'),
                            row=row, col=col
                        )

                fig_yearly.update_layout(
                    title="Yearly Precipitation Analysis with Trends",
                    height=600,
                    showlegend=False
                )

                # Update x-axis labels
                fig_yearly.update_xaxes(title_text="Year")
                fig_yearly.update_yaxes(title_text="Precipitation (mm)", row=1, col=1)
                fig_yearly.update_yaxes(title_text="Precipitation (mm/day)", row=1, col=2)
                fig_yearly.update_yaxes(title_text="Precipitation (mm)", row=2, col=1)
                fig_yearly.update_yaxes(title_text="Precipitation (mm/day)", row=2, col=2)

                st.plotly_chart(fig_yearly, use_container_width=True)

                # Summary of trends
                trend_summary = []
                for metric, (_, ylabel, _, _) in zip(['max', 'mean', 'total', 'median'], metrics):
                    if metric in trends:
                        trend_info = trends[metric]
                        trend_summary.append(f"**{ylabel.split('(')[0].strip()}:** {trend_info['direction']} {trend_info['trend']}")

                if trend_summary:
                    st.info("**Trend Summary:**\n" + " | ".join(trend_summary))

            else:
                st.info("💡 Yearly analysis requires at least one full year of data. Please try a longer date range.")

        except Exception as e:
            st.error(f"❌ Error creating yearly chart: {str(e)}")
            st.info("💡 Try refreshing the data or selecting a longer date range.")
    else:
        st.info("💡 Yearly analysis not available. Please refresh the data.")


def _render_statistical_analysis():
    """Render statistical analysis"""
    st.markdown("### 📊 Statistical Summary")

    precipitation_data = st.session_state.hydro_precipitation_data
    analyzer = st.session_state.get('hydro_analyzer')
    precip_values = precipitation_data['precipitation']

    # Basic daily statistics
    st.markdown("#### Daily Statistics")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Daily Mean", f"{precip_values.mean():.2f} mm")
        st.metric("Daily Median", f"{precip_values.median():.2f} mm")

    with col2:
        st.metric("Daily Maximum", f"{precip_values.max():.2f} mm")
        st.metric("Daily Minimum", f"{precip_values.min():.2f} mm")

    with col3:
        st.metric("Std Dev", f"{precip_values.std():.2f} mm")
        st.metric("Total Days", f"{len(precip_values)}")

    with col4:
        wet_days = (precip_values > 1.0).sum()
        st.metric("Wet Days", f"{wet_days}")
        st.metric("Wet Day %", f"{wet_days/len(precip_values)*100:.1f}%")

    # Yearly statistics with trends
    if analyzer:
        try:
            yearly_stats = analyzer.calculate_yearly_statistics()
        except AttributeError as e:
            # Handle case where analyzer doesn't have the new method
            st.info("🔄 Updating analyzer with new features...")
            try:
                from geoclimate_fetcher.hydrology_analysis import HydrologyAnalyzer
                import importlib
                import geoclimate_fetcher.hydrology_analysis

                # Reload the module to get the latest version
                importlib.reload(geoclimate_fetcher.hydrology_analysis)

                # Create a new analyzer with updated methods
                new_analyzer = HydrologyAnalyzer(st.session_state.hydro_geometry)
                new_analyzer.precipitation_data = precipitation_data.copy()

                # Verify the method exists before using it
                if hasattr(new_analyzer, 'calculate_yearly_statistics'):
                    st.session_state.hydro_analyzer = new_analyzer
                    yearly_stats = new_analyzer.calculate_yearly_statistics()
                    st.success("✅ Analyzer updated successfully with yearly statistics!")
                else:
                    st.error("❌ Could not load updated analyzer. Please refresh the page.")
                    yearly_stats = None

            except Exception as update_error:
                st.error(f"❌ Failed to update analyzer: {str(update_error)}")
                st.info("💡 Please refresh the data to access enhanced yearly statistics and trends.")
                yearly_stats = None
        except Exception as e:
            st.error(f"❌ Error calculating yearly statistics: {str(e)}")
            yearly_stats = None

        if yearly_stats and 'yearly_data' in yearly_stats and len(yearly_stats['yearly_data']) > 0:
            st.markdown("#### Yearly Statistics & Trends")

            # Summary metrics with trends
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                trend_info = yearly_stats['trends'].get('max', {})
                direction = trend_info.get('direction', '❓')
                trend_text = trend_info.get('trend', 'unknown')
                st.metric(
                    f"Yearly Max {direction}",
                    f"{yearly_stats['summary']['max_value']:.1f} mm",
                    delta=f"Trend: {trend_text}"
                )

            with col2:
                trend_info = yearly_stats['trends'].get('mean', {})
                direction = trend_info.get('direction', '❓')
                trend_text = trend_info.get('trend', 'unknown')
                mean_val = yearly_stats['yearly_data']['mean'].mean()
                st.metric(
                    f"Yearly Mean {direction}",
                    f"{mean_val:.2f} mm/day",
                    delta=f"Trend: {trend_text}"
                )

            with col3:
                trend_info = yearly_stats['trends'].get('median', {})
                direction = trend_info.get('direction', '❓')
                trend_text = trend_info.get('trend', 'unknown')
                median_val = yearly_stats['yearly_data']['median'].mean()
                st.metric(
                    f"Yearly Median {direction}",
                    f"{median_val:.2f} mm/day",
                    delta=f"Trend: {trend_text}"
                )

            with col4:
                trend_info = yearly_stats['trends'].get('total', {})
                direction = trend_info.get('direction', '❓')
                trend_text = trend_info.get('trend', 'unknown')
                st.metric(
                    f"Annual Total {direction}",
                    f"{yearly_stats['summary']['mean_annual_total']:.0f} mm",
                    delta=f"Trend: {trend_text}"
                )

            # Show yearly data table
            with st.expander("📊 Yearly Data Table", expanded=False):
                yearly_df = yearly_stats['yearly_data'].copy()
                yearly_df = yearly_df.round(2)
                st.dataframe(yearly_df, use_container_width=True, hide_index=True)

            # Show key insights
            summary = yearly_stats['summary']
            st.info(f"""
            **📊 Key Insights:**
            - **Analysis Period:** {summary['years_analyzed']} years ({summary['start_year']}-{summary['end_year']})
            - **Wettest Year:** {summary['wettest_year']} ({yearly_stats['yearly_data'].loc[yearly_stats['yearly_data']['year'] == summary['wettest_year'], 'total'].iloc[0]:.0f} mm)
            - **Driest Year:** {summary['driest_year']} ({yearly_stats['yearly_data'].loc[yearly_stats['yearly_data']['year'] == summary['driest_year'], 'total'].iloc[0]:.0f} mm)
            - **Highest Daily Maximum:** {summary['max_value']:.1f} mm in {summary['max_year']}
            """)

        else:
            # Show fallback message when yearly analysis is not available
            st.markdown("#### ℹ️ Yearly Analysis Not Available")

            if analyzer and hasattr(analyzer, 'precipitation_data') and analyzer.precipitation_data is not None:
                data_info = []
                try:
                    num_rows = len(analyzer.precipitation_data)
                    date_range = (analyzer.precipitation_data['date'].max() - analyzer.precipitation_data['date'].min()).days
                    years = date_range / 365.25

                    data_info.append(f"**Data Available:** {num_rows:,} daily records")
                    data_info.append(f"**Time Span:** {years:.1f} years")

                    if years < 1:
                        data_info.append("**Issue:** Less than 1 full year of data")
                        data_info.append("**Recommendation:** Try a longer date range to enable yearly analysis")
                    elif years < 3:
                        data_info.append("**Issue:** Less than 3 years of data")
                        data_info.append("**Recommendation:** Trend analysis requires at least 3 years of data")
                    else:
                        data_info.append("**Issue:** Data validation or processing error")
                        data_info.append("**Recommendation:** Try refreshing the data")

                except Exception:
                    data_info.append("**Issue:** Unable to analyze data structure")
                    data_info.append("**Recommendation:** Please refresh the data")

                st.info("\n".join(data_info))
            else:
                st.info("""
                **Yearly statistics and trends are not available.**

                **Possible reasons:**
                - Insufficient data (need at least 1 full year)
                - Data format issues
                - Analysis errors

                **Try:**
                - Refreshing the data with the button above
                - Selecting a longer date range
                - Choosing a different dataset
                """)

    # Distribution plot
    st.markdown("#### 📊 Precipitation Distribution")

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=precip_values[precip_values > 0],  # Only wet days
        nbinsx=50,
        name='Precipitation Distribution',
        marker_color='skyblue'
    ))

    fig.update_layout(
        title="Distribution of Daily Precipitation (Wet Days Only)",
        xaxis_title="Precipitation (mm)",
        yaxis_title="Frequency",
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)


def _render_return_period_analysis():
    """Render return period analysis using HydrologyAnalyzer methods"""
    st.markdown("### 🌧️ Return Period Analysis")

    try:
        # Get the HydrologyAnalyzer instance
        analyzer = st.session_state.get('hydro_analyzer')
        if analyzer is None:
            st.error("❌ Analyzer not available. Please fetch data first.")
            return

        with st.spinner("🔄 Calculating return periods using multiple distributions..."):
            # Calculate annual maxima using the analyzer
            annual_maxima = analyzer.calculate_annual_maxima()

            if annual_maxima.empty or len(annual_maxima) < 3:
                st.warning("⚠️ Insufficient data for reliable return period analysis. Need at least 3 years of data.")
                st.info(f"Currently have {len(annual_maxima)} years of annual maximum data")
                return

            # Calculate return periods using multiple distributions
            return_analysis = analyzer.calculate_return_periods(annual_maxima)

            if not return_analysis or 'distributions' not in return_analysis:
                st.error("❌ Unable to calculate return periods")
                return

        st.success(f"✅ Return period analysis completed using {len(annual_maxima)} years of annual maxima")

        # Display return period results for different distributions
        st.markdown("#### 📊 Multiple Distribution Analysis")

        # Get the best fitting distribution (lowest KS statistic)
        best_dist = None
        lowest_ks = float('inf')

        for dist_name, dist_data in return_analysis['distributions'].items():
            if 'ks_statistic' in dist_data and dist_data['ks_statistic'] < lowest_ks:
                lowest_ks = dist_data['ks_statistic']
                best_dist = dist_name

        if best_dist:
            st.info(f"🎯 Best fitting distribution: **{best_dist}** (KS statistic: {lowest_ks:.4f})")

        # Display return period table for best distribution
        if best_dist and 'return_values' in return_analysis['distributions'][best_dist]:
            st.markdown(f"#### 📋 Return Period Values - {best_dist} Distribution")

            return_periods = return_analysis['return_periods']
            return_values = return_analysis['distributions'][best_dist]['return_values']

            return_df = pd.DataFrame({
                'Return Period (years)': return_periods,
                'Precipitation (mm)': [f"{val:.1f}" for val in return_values],
                'Probability (%)': [f"{100/rp:.2f}" for rp in return_periods]
            })

            st.dataframe(return_df, use_container_width=True, hide_index=True)

            # Analyze data for outlier detection and plot preparation
            all_values = []
            valid_distributions = {}

            for dist_name, dist_data in return_analysis['distributions'].items():
                if 'return_values' in dist_data:
                    values = np.array(dist_data['return_values'])
                    # Check for extreme outliers or invalid values
                    if np.all(np.isfinite(values)) and np.all(values > 0):
                        max_reasonable = np.percentile(annual_maxima['annual_max_precipitation'], 99.9) * 10
                        if np.max(values) < max_reasonable:  # Filter out extremely large values
                            valid_distributions[dist_name] = dist_data
                            all_values.extend(values)

            if not valid_distributions:
                st.warning("⚠️ All distributions produced extreme values. Using best distribution only.")
                valid_distributions = {best_dist: return_analysis['distributions'][best_dist]}

            # Calculate reasonable y-axis limits
            if all_values:
                y_min = min(all_values) * 0.8
                y_max = max(all_values) * 1.2
                # Limit extreme upper bounds
                max_observed = annual_maxima['annual_max_precipitation'].max()
                if y_max > max_observed * 20:  # If predicted values are too extreme
                    y_max = max_observed * 10
            else:
                y_min, y_max = 0, 100

            # Create visualization options
            show_all = st.checkbox("Show all valid distributions",
                                 value=len(valid_distributions) <= 3,
                                 help="Uncheck to show only the best-fitting distribution")

            # Plot return period curves
            fig = go.Figure()
            colors = ['red', 'blue', 'green', 'purple', 'orange']

            distributions_to_show = valid_distributions if show_all else {best_dist: valid_distributions.get(best_dist, {})}

            for i, (dist_name, dist_data) in enumerate(distributions_to_show.items()):
                if 'return_values' in dist_data:
                    line_style = dict(width=3) if dist_name == best_dist else dict(width=2, dash='dash')
                    opacity = 1.0 if dist_name == best_dist else 0.7

                    fig.add_trace(go.Scatter(
                        x=return_periods,
                        y=dist_data['return_values'],
                        mode='lines+markers',
                        name=f'{dist_name}' + (' (Best)' if dist_name == best_dist else ''),
                        line=dict(color=colors[i % len(colors)], **line_style),
                        opacity=opacity,
                        marker=dict(size=6 if dist_name == best_dist else 4)
                    ))

            # Add observed data points for reference
            if not annual_maxima.empty:
                sorted_maxima = np.sort(annual_maxima['annual_max_precipitation'])[::-1]  # Descending order
                n = len(sorted_maxima)
                empirical_rp = [(n + 1) / (i + 1) for i in range(min(n, 10))]  # Show top 10
                empirical_values = sorted_maxima[:min(n, 10)]

                fig.add_trace(go.Scatter(
                    x=empirical_rp,
                    y=empirical_values,
                    mode='markers',
                    name='Observed Data',
                    marker=dict(size=8, color='black', symbol='circle-open'),
                    opacity=0.8
                ))

            fig.update_layout(
                title="Precipitation Return Period Curves",
                xaxis_title="Return Period (years)",
                yaxis_title="Precipitation (mm)",
                xaxis_type="log",
                yaxis_range=[y_min, y_max],  # Set reasonable y-axis limits
                height=500,
                showlegend=True
            )

            st.plotly_chart(fig, use_container_width=True)

            # Show filtering info if distributions were removed
            removed_count = len(return_analysis['distributions']) - len(valid_distributions)
            if removed_count > 0:
                st.info(f"ℹ️ {removed_count} distribution(s) filtered out due to extreme predictions")

            # Show distribution comparison
            st.markdown("#### 📈 Distribution Goodness-of-Fit")

            fit_data = []
            for dist_name, dist_data in return_analysis['distributions'].items():
                if 'ks_statistic' in dist_data:
                    fit_data.append({
                        'Distribution': dist_name,
                        'KS Statistic': f"{dist_data['ks_statistic']:.4f}",
                        'P-value': f"{dist_data.get('ks_pvalue', 0):.4f}",
                        'Quality': 'Excellent' if dist_data['ks_statistic'] < 0.1 else
                                  'Good' if dist_data['ks_statistic'] < 0.2 else 'Fair'
                    })

            if fit_data:
                fit_df = pd.DataFrame(fit_data)
                st.dataframe(fit_df, use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"❌ Error in return period analysis: {str(e)}")
        st.info("This might be due to insufficient data or computational issues with extreme value fitting.")
    
    # Interpretation
    st.markdown("#### 💡 Interpretation")
    st.info(f"""
    **Key Insights:**
    - **2-year return period**: {return_values[0]:.1f} mm (50% chance annually)
    - **10-year return period**: {return_values[2]:.1f} mm (10% chance annually)
    - **100-year return period**: {return_values[5]:.1f} mm (1% chance annually)
    
    These values represent the expected maximum daily precipitation for different return periods.
    """)


def _render_educational_content():
    """Render educational content"""
    st.markdown("### 📚 Hydrology Concepts Explained")
    
    with st.expander("🎓 Return Period Analysis", expanded=True):
        st.markdown("""
        **Definition**: The average time interval between events of a given magnitude
        - **Application**: Flood risk assessment, infrastructure design
        - **Key Point**: A 100-year event has a 1% chance of occurring in any given year
        - **Common Misconception**: It doesn't mean the event occurs exactly every 100 years
        """)
    
    with st.expander("📊 Statistical Distributions"):
        st.markdown("""
        **Extreme Value Distributions**:
        - **Gumbel**: Commonly used for annual maximum precipitation
        - **Log-Normal**: Often fits precipitation data well
        - **Generalized Extreme Value (GEV)**: More flexible, includes Gumbel as special case
        
        **Fitting Methods**:
        - Method of Moments
        - Maximum Likelihood Estimation
        - L-Moments (more robust)
        """)
    
    with st.expander("🌧️ Precipitation Indices"):
        st.markdown("""
        **Common Precipitation Indices**:
        - **PRCPTOT**: Annual total precipitation
        - **R10mm**: Number of days with precipitation ≥ 10mm
        - **R95p**: Total precipitation from very wet days (>95th percentile)
        - **CDD**: Maximum consecutive dry days
        - **CWD**: Maximum consecutive wet days
        
        **Time Scales**: 3, 6, 12 months capture different drought impacts
        """)
    
    with st.expander("📖 Further Learning"):
        st.markdown("""
        **Recommended Resources**:
        - [ETCCDI Climate Indices](http://etccdi.pacificclimate.org/)
        - [WMO Guidelines on Extreme Weather](https://www.wmo.int/)
        - Engineering Hydrology textbooks for detailed theory
        - [USGS Water Resources](https://www.usgs.gov/mission-areas/water-resources)
        """)


def _render_sample_results():
    """Render sample results for demonstration"""
    st.markdown("## 📊 Sample Hydrology Results")
    st.info("This is a demonstration of the analysis output format. Run a real analysis with your data above.")
    
    # Generate sample data for demonstration
    dates = pd.date_range('2019-01-01', '2023-12-31', freq='D')
    sample_precip = np.random.exponential(2.5, len(dates))
    sample_precip[np.random.random(len(dates)) > 0.7] = 0  # Make some days dry
    
    sample_data = pd.DataFrame({
        'date': dates,
        'precipitation': sample_precip
    })
    
    # Store sample data temporarily
    original_data = st.session_state.get('hydro_precipitation_data')
    st.session_state.hydro_precipitation_data = sample_data
    
    # Render the same results with sample data
    tab1, tab2, tab3 = st.tabs(["📈 Time Series", "📊 Statistics", "🌧️ Return Periods"])
    
    with tab1:
        _render_time_series_analysis()
    
    with tab2:
        _render_statistical_analysis()
    
    with tab3:
        _render_return_period_analysis()
    
    # Restore original data
    if original_data is not None:
        st.session_state.hydro_precipitation_data = original_data
    else:
        del st.session_state.hydro_precipitation_data


def _get_precipitation_datasets():
    """Get available precipitation datasets"""
    return [
        {
            'name': 'CHIRPS Daily',
            'provider': 'UCSB Climate Hazards Group',
            'resolution': '0.05° (~5.5 km)',
            'period': '1981-present',
            'description': 'High-resolution precipitation dataset combining satellite and station data',
            'ee_id': 'UCSB-CHG/CHIRPS/DAILY',
            'precipitation_band': 'precipitation',
            'unit': 'mm',
            'start_date': '1981-01-01',
            'end_date': '2025-09-12'
        },
        {
            'name': 'GPM IMERG V07',
            'provider': 'NASA',
            'resolution': '0.1° (~11 km)',
            'period': '2000-06-01 to present',
            'description': 'GPM IMERG V07 - snapshot precipitation (calibrated)',
            'ee_id': 'NASA/GPM_L3/IMERG_V07',
            'precipitation_band': 'precipitation',
            'unit': 'mm/hr',
            'start_date': '2000-06-01',
            'end_date': '2025-09-12'
        },
        {
            'name': 'ERA5-Land Daily',
            'provider': 'ECMWF',
            'resolution': '0.1° (~11 km)',
            'period': '1950-01-02 to present',
            'description': 'ERA5-Land Daily Aggregated - ECMWF Climate Reanalysis',
            'ee_id': 'ECMWF/ERA5_LAND/DAILY_AGGR',
            'precipitation_band': 'total_precipitation_sum',
            'unit': 'm',
            'start_date': '1950-01-02',
            'end_date': '2025-09-06'
        }
    ]


def _simulate_precipitation_data(start_date, end_date):
    """Simulate precipitation data for demonstration"""
    dates = pd.date_range(start_date, end_date, freq='D')
    
    # Generate realistic precipitation data
    # Use exponential distribution with seasonal variation
    day_of_year = dates.dayofyear
    seasonal_factor = 1 + 0.5 * np.sin(2 * np.pi * (day_of_year - 90) / 365)
    
    precipitation = np.random.exponential(2.0 * seasonal_factor, len(dates))
    
    # Make some days dry (about 70% of days)
    dry_days = np.random.random(len(dates)) > 0.3
    precipitation[dry_days] = 0
    
    return pd.DataFrame({
        'date': dates,
        'precipitation': precipitation
    })


def _calculate_monthly_stats(precipitation_data):
    """Calculate monthly precipitation statistics"""
    monthly = precipitation_data.groupby(precipitation_data['date'].dt.month).agg({
        'precipitation': ['sum', 'mean', 'count']
    }).round(2)
    
    monthly.columns = ['total_precip', 'mean_precip', 'days']
    monthly['month'] = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][:len(monthly)]
    
    return monthly.reset_index(drop=True)


def _calculate_annual_maxima(precipitation_data):
    """Calculate annual maximum precipitation"""
    annual_max = precipitation_data.groupby(precipitation_data['date'].dt.year)['precipitation'].max()
    return annual_max.values


def _calculate_return_values(annual_maxima, return_periods):
    """Calculate return period values using Gumbel distribution"""
    from scipy import stats
    
    # Fit Gumbel distribution
    params = stats.gumbel_r.fit(annual_maxima)
    
    # Calculate return values
    probabilities = [1 - 1/rp for rp in return_periods]
    return_values = stats.gumbel_r.ppf(probabilities, *params)
    
    return return_values


def _render_hydrology_downloads():
    """Render download options for hydrology analysis results"""
    st.markdown("### 💾 Download Analysis Results")

    # Check if we have data to download
    precipitation_data = st.session_state.get('hydro_precipitation_data')
    analyzer = st.session_state.get('hydro_analyzer')

    if precipitation_data is None or analyzer is None:
        st.warning("⚠️ No analysis data available for download. Please run the analysis first.")
        return

    st.info("📊 Available downloads from your hydrology analysis:")

    # Create download sections
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📈 Raw Precipitation Data")

        # Raw precipitation data CSV
        if precipitation_data is not None and not precipitation_data.empty:
            csv_data = precipitation_data.to_csv(index=False)
            dataset_name = st.session_state.get('hydro_dataset_info', {}).get('name', 'precipitation')
            filename = f"{dataset_name.replace(' ', '_')}_raw_data.csv"

            st.download_button(
                label=f"📥 Download Raw Data CSV ({len(csv_data)/1024:.1f} KB)",
                data=csv_data,
                file_name=filename,
                mime="text/csv",
                help="Download the raw daily precipitation data used in the analysis"
            )

            st.caption(f"Contains {len(precipitation_data)} days of precipitation data")

    with col2:
        st.markdown("#### 📊 Yearly Statistics & Trends")

        try:
            # Get yearly statistics with trends
            try:
                yearly_stats = analyzer.calculate_yearly_statistics()
            except AttributeError:
                # Handle case where analyzer doesn't have the new method
                try:
                    from geoclimate_fetcher.hydrology_analysis import HydrologyAnalyzer
                    import importlib
                    import geoclimate_fetcher.hydrology_analysis

                    # Reload the module to get the latest version
                    importlib.reload(geoclimate_fetcher.hydrology_analysis)

                    # Create a new analyzer with updated methods
                    new_analyzer = HydrologyAnalyzer(st.session_state.hydro_geometry)
                    new_analyzer.precipitation_data = precipitation_data.copy()

                    # Verify the method exists before using it
                    if hasattr(new_analyzer, 'calculate_yearly_statistics'):
                        st.session_state.hydro_analyzer = new_analyzer
                        yearly_stats = new_analyzer.calculate_yearly_statistics()
                    else:
                        yearly_stats = None
                except Exception:
                    yearly_stats = None
            except Exception as e:
                st.error(f"Error calculating yearly statistics: {str(e)}")
                yearly_stats = None

            if yearly_stats and 'yearly_data' in yearly_stats:
                yearly_df = yearly_stats['yearly_data'].copy()

                # Add trend information
                trends = yearly_stats['trends']
                for metric in ['max', 'mean', 'median', 'total']:
                    if metric in trends:
                        trend_info = trends[metric]
                        yearly_df[f'{metric}_trend'] = trend_info['trend']
                        yearly_df[f'{metric}_slope'] = round(trend_info['slope'], 6)

                yearly_csv = yearly_df.to_csv(index=False)
                yearly_filename = f"{dataset_name.replace(' ', '_')}_yearly_statistics.csv"

                st.download_button(
                    label=f"📥 Download Yearly Stats CSV ({len(yearly_csv)/1024:.1f} KB)",
                    data=yearly_csv,
                    file_name=yearly_filename,
                    mime="text/csv",
                    help="Download yearly max, mean, median with trend analysis"
                )

                st.caption(f"Contains {len(yearly_df)} years of statistics and trends")

            else:
                # Fallback to basic stats if yearly calculation fails
                stats_data = {
                    'Metric': [
                        'Mean Precipitation (mm/day)',
                        'Median Precipitation (mm/day)',
                        'Maximum Precipitation (mm)',
                        'Standard Deviation (mm)',
                        'Total Precipitation (mm)',
                        'Wet Days Count',
                        'Dry Days Count',
                        'Number of Records'
                    ],
                    'Value': [
                        round(precipitation_data['precipitation'].mean(), 2),
                        round(precipitation_data['precipitation'].median(), 2),
                        round(precipitation_data['precipitation'].max(), 2),
                        round(precipitation_data['precipitation'].std(), 2),
                        round(precipitation_data['precipitation'].sum(), 2),
                        len(precipitation_data[precipitation_data['precipitation'] > 0]),
                        len(precipitation_data[precipitation_data['precipitation'] == 0]),
                        len(precipitation_data)
                    ]
                }

                stats_df = pd.DataFrame(stats_data)
                stats_csv = stats_df.to_csv(index=False)
                stats_filename = f"{dataset_name.replace(' ', '_')}_statistics.csv"

                st.download_button(
                    label=f"📥 Download Statistics CSV ({len(stats_csv)/1024:.1f} KB)",
                    data=stats_csv,
                    file_name=stats_filename,
                    mime="text/csv",
                    help="Download summary statistics of the precipitation data"
                )

                st.caption("Contains basic statistical metrics")

        except Exception as e:
            st.error(f"Error generating statistics: {str(e)}")

    # Return period analysis download
    st.markdown("#### 🌧️ Return Period Analysis")

    try:
        # Get return period data from analyzer
        annual_maxima = _calculate_annual_maxima(precipitation_data)
        return_periods = [2, 5, 10, 20, 50, 100]
        return_values = _calculate_return_values(annual_maxima, return_periods)

        # Create return period DataFrame
        return_df = pd.DataFrame({
            'Return_Period_Years': return_periods,
            'Expected_Precipitation_mm': [round(val, 2) for val in return_values]
        })

        # Add annual maxima data
        maxima_df = pd.DataFrame({
            'Year': range(len(annual_maxima)),
            'Annual_Maximum_Precipitation_mm': [round(val, 2) for val in annual_maxima]
        })

        col3, col4 = st.columns(2)

        with col3:
            return_csv = return_df.to_csv(index=False)
            return_filename = f"{dataset_name.replace(' ', '_')}_return_periods.csv"

            st.download_button(
                label=f"📥 Download Return Periods CSV ({len(return_csv)/1024:.1f} KB)",
                data=return_csv,
                file_name=return_filename,
                mime="text/csv",
                help="Download return period analysis results"
            )

        with col4:
            maxima_csv = maxima_df.to_csv(index=False)
            maxima_filename = f"{dataset_name.replace(' ', '_')}_annual_maxima.csv"

            st.download_button(
                label=f"📥 Download Annual Maxima CSV ({len(maxima_csv)/1024:.1f} KB)",
                data=maxima_csv,
                file_name=maxima_filename,
                mime="text/csv",
                help="Download annual maximum precipitation values"
            )

    except Exception as e:
        st.warning(f"⚠️ Return period data not available: {str(e)}")

    # Add usage information
    with st.expander("💡 How to use downloaded data"):
        st.markdown("""
        **Raw Precipitation Data:**
        - Contains daily precipitation values for your selected area and time period
        - Use in spreadsheet software (Excel, Google Sheets) or programming languages (Python, R)
        - Columns: date, precipitation (mm/day)

        **Statistical Analysis:**
        - Summary statistics for quick reference
        - Use for reports, presentations, or further analysis

        **Return Period Analysis:**
        - Expected precipitation values for different return periods (2, 5, 10, 20, 50, 100 years)
        - Critical for flood risk assessment and infrastructure design
        - Annual maxima data shows the highest precipitation recorded each year

        **File Formats:**
        - All files are in CSV format for maximum compatibility
        - Can be opened in Excel, Google Sheets, Python pandas, R, MATLAB, etc.
        """)


def _render_idf_analysis():
    """Render enhanced IDF curves analysis"""
    st.markdown("### 📉 Intensity-Duration-Frequency (IDF) Curves")

    st.info("""
    **What are IDF curves?**
    IDF curves show the relationship between rainfall intensity, duration, and return period.
    They're essential for designing drainage systems, culverts, and flood protection infrastructure.
    """)

    analyzer = st.session_state.hydro_analyzer

    with st.spinner("🔄 Calculating IDF curves..."):
        try:
            # Get dates from session state
            start_date_str = st.session_state.hydro_start_date.strftime('%Y-%m-%d')
            end_date_str = st.session_state.hydro_end_date.strftime('%Y-%m-%d')

            # Calculate IDF (will automatically use server-side for sub-daily data)
            idf_results = analyzer.calculate_intensity_duration_frequency(
                start_date=start_date_str,
                end_date=end_date_str,
                return_periods=[2, 5, 10, 25, 50, 100]
            )

            if idf_results:
                # Create and display plot
                idf_plot = analyzer.create_idf_curves_plot(idf_results)
                st.plotly_chart(idf_plot, use_container_width=True)

                # Show detailed results table
                st.markdown("#### 📊 IDF Values Table")

                # Create summary table
                idf_table_data = []
                for duration_key, data in idf_results.items():
                    duration_hrs = data.get('duration_hours', duration_key)
                    return_periods = data.get('return_periods', [])
                    intensities = data.get('return_intensities', data.get('return_intensities_mm_per_hour', []))

                    row = {'Duration (hours)': duration_hrs}
                    for rp, intensity in zip(return_periods, intensities):
                        row[f'{rp}-year (mm/hr)'] = f"{intensity:.2f}"

                    idf_table_data.append(row)

                idf_df = pd.DataFrame(idf_table_data)
                st.dataframe(idf_df, use_container_width=True)

                # Educational content
                with st.expander("📚 Understanding IDF Curves"):
                    st.markdown("""
                    **How to Read IDF Curves:**
                    - **X-axis (Duration)**: Time period over which rainfall occurs (e.g., 1 hour, 3 hours, 24 hours)
                    - **Y-axis (Intensity)**: Average rainfall rate in mm/hr
                    - **Each line**: Represents a different return period (e.g., 10-year, 50-year storm)

                    **Example Interpretation:**
                    - A 50-year, 3-hour storm might have an intensity of 15 mm/hr
                    - This means: In a 3-hour period, we expect 15 × 3 = 45 mm of rain
                    - This event has a 2% chance of occurring in any given year (1/50 = 0.02)

                    **Applications:**
                    - Storm sewer design
                    - Culvert sizing
                    - Detention pond design
                    - Urban drainage planning
                    """)

            else:
                st.warning("⚠️ Insufficient data for IDF analysis. Need at least 3 years of data.")

        except Exception as e:
            st.error(f"❌ Error calculating IDF curves: {str(e)}")
            st.info("This might occur if the dataset doesn't support sub-daily analysis.")


def _render_mann_kendall_analysis():
    """Render Mann-Kendall trend analysis"""
    st.markdown("### 📈 Mann-Kendall Trend Analysis")

    st.info("""
    **What is Mann-Kendall Test?**
    A non-parametric statistical test for detecting monotonic trends in time series data.
    More robust than linear regression because it doesn't assume normal distribution.
    """)

    analyzer = st.session_state.hydro_analyzer

    # Aggregation selection
    aggregation = st.radio(
        "Select temporal aggregation:",
        options=['annual', 'monthly', 'seasonal'],
        format_func=lambda x: x.capitalize(),
        horizontal=True,
        help="Annual: Tests trend in yearly totals | Monthly: Tests trend in monthly data | Seasonal: Tests each season separately"
    )

    if st.button("🔍 Run Mann-Kendall Test", type="primary"):
        with st.spinner(f"🔄 Running Mann-Kendall test on {aggregation} data..."):
            try:
                mk_results = analyzer.calculate_mann_kendall_trends(aggregation=aggregation)

                if 'error' in mk_results:
                    st.error(f"❌ {mk_results['error']}")
                else:
                    # Display interpretation
                    st.markdown("#### 📊 Test Results")

                    if aggregation in ['annual', 'monthly']:
                        # Main results
                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            trend = mk_results.get('trend', 'N/A')
                            trend_emoji = "📈" if trend == 'increasing' else "📉" if trend == 'decreasing' else "➡️"
                            st.metric("Trend", f"{trend_emoji} {trend.capitalize()}")

                        with col2:
                            p_value = mk_results.get('p_value', 0)
                            sig = "✅ Significant" if p_value < 0.05 else "⚠️ Not Significant"
                            st.metric("Significance (α=0.05)", sig)

                        with col3:
                            slope = mk_results.get('slope', 0)
                            slope_unit = mk_results.get('slope_unit', 'mm/yr')
                            st.metric("Sen's Slope", f"{slope:.2f} {slope_unit}")

                        with col4:
                            tau = mk_results.get('tau', 0)
                            st.metric("Kendall's Tau", f"{tau:.4f}")

                        # Interpretation
                        st.success(f"**Interpretation:** {mk_results.get('interpretation', '')}")

                        # Statistical details
                        with st.expander("📊 Statistical Details"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("p-value", f"{mk_results.get('p_value', 0):.6f}")
                                st.metric("Series Length", mk_results.get('series_length', 0))
                            with col2:
                                st.metric("z-score", f"{mk_results.get('z_score', 0):.4f}")

                        # Visualization
                        mk_plot = analyzer.create_mann_kendall_plot(mk_results)
                        st.plotly_chart(mk_plot, use_container_width=True)

                    elif aggregation == 'seasonal':
                        # Seasonal trends
                        seasonal_trends = mk_results.get('seasonal_trends', {})

                        st.markdown("#### Seasonal Trend Summary")
                        for season, results in seasonal_trends.items():
                            trend = results.get('trend', 'N/A')
                            p_value = results.get('p_value', 1)
                            slope = results.get('slope', 0)
                            sig = results.get('significance', 'not significant')

                            trend_emoji = "📈" if trend == 'increasing' else "📉" if trend == 'decreasing' else "➡️"
                            sig_emoji = "✅" if sig == 'significant' else "⚠️"

                            st.markdown(f"""
                            **{season}:** {trend_emoji} {trend.capitalize()} {sig_emoji} ({sig})
                            Sen's Slope: {slope:.2f} mm/year | p-value: {p_value:.4f}
                            """)

                    # Educational content
                    with st.expander("📚 Understanding Mann-Kendall Results"):
                        st.markdown("""
                        **Key Concepts:**

                        **Trend Direction:**
                        - **Increasing**: Values tend to increase over time
                        - **Decreasing**: Values tend to decrease over time
                        - **No trend**: No consistent monotonic pattern

                        **Statistical Significance (p-value):**
                        - p < 0.05: Statistically significant (95% confidence)
                        - p < 0.01: Highly significant (99% confidence)
                        - p ≥ 0.05: Not statistically significant

                        **Sen's Slope:**
                        - Median of all pairwise slopes
                        - Magnitude of the trend (mm/year or mm/month)
                        - Robust to outliers

                        **Kendall's Tau (τ):**
                        - Correlation coefficient between -1 and 1
                        - τ > 0: Positive correlation (increasing trend)
                        - τ < 0: Negative correlation (decreasing trend)
                        - |τ| close to 1: Strong monotonic relationship
                        """)

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")


def _render_pettitt_analysis():
    """Render Pettitt change point analysis"""
    st.markdown("### 🔄 Pettitt Change Point Test")

    st.info("""
    **What is Pettitt Test?**
    A non-parametric test for detecting a single change point in a time series.
    Identifies when the statistical properties of precipitation changed significantly.
    """)

    analyzer = st.session_state.hydro_analyzer

    # Aggregation selection
    aggregation = st.radio(
        "Select temporal aggregation:",
        options=['annual', 'monthly'],
        format_func=lambda x: x.capitalize(),
        horizontal=True,
        key="pettitt_aggregation"
    )

    if st.button("🔍 Run Pettitt Test", type="primary"):
        with st.spinner(f"🔄 Running Pettitt test on {aggregation} data..."):
            try:
                pettitt_results = analyzer.calculate_pettitt_test(aggregation=aggregation)

                if 'error' in pettitt_results:
                    st.error(f"❌ {pettitt_results['error']}")
                else:
                    # Display interpretation
                    st.markdown("#### 📊 Change Point Detection Results")

                    is_significant = pettitt_results.get('is_significant', False)

                    if is_significant:
                        st.success(f"**✅ {pettitt_results.get('interpretation', '')}**")
                    else:
                        st.info(f"**ℹ️ {pettitt_results.get('interpretation', '')}**")

                    # Main metrics
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        change_date = pettitt_results.get('change_point_date', 'N/A')
                        st.metric("Change Point", change_date)

                    with col2:
                        p_value = pettitt_results.get('p_value', 0)
                        sig = "✅ Significant" if p_value < 0.05 else "⚠️ Not Significant"
                        st.metric("Significance", sig)

                    with col3:
                        change_mag = pettitt_results.get('change_magnitude_percent', 0)
                        direction = "📈" if change_mag > 0 else "📉"
                        st.metric("Change Magnitude", f"{direction} {abs(change_mag):.1f}%")

                    with col4:
                        u_stat = pettitt_results.get('U_statistic', 0)
                        st.metric("U Statistic", f"{u_stat:.1f}")

                    # Before/After comparison
                    st.markdown("#### 📊 Before vs After Comparison")
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("**Before Change Point:**")
                        mean_before = pettitt_results.get('mean_before', 0)
                        std_before = pettitt_results.get('std_before', 0)
                        st.metric("Mean Precipitation", f"{mean_before:.1f} mm")
                        st.metric("Std Deviation", f"{std_before:.1f} mm")

                    with col2:
                        st.markdown("**After Change Point:**")
                        mean_after = pettitt_results.get('mean_after', 0)
                        std_after = pettitt_results.get('std_after', 0)
                        st.metric("Mean Precipitation", f"{mean_after:.1f} mm")
                        st.metric("Std Deviation", f"{std_after:.1f} mm")

                    # Visualization
                    pettitt_plot = analyzer.create_pettitt_plot(pettitt_results)
                    st.plotly_chart(pettitt_plot, use_container_width=True)

                    # Educational content
                    with st.expander("📚 Understanding Pettitt Test"):
                        st.markdown("""
                        **What does a change point mean?**

                        A change point indicates a significant shift in the statistical properties of the time series.
                        This could be due to:
                        - Climate change effects
                        - Land use changes (urbanization, deforestation)
                        - Changes in measurement methods
                        - Natural climate variability (regime shifts)

                        **How to interpret:**
                        - **Significant change (p < 0.05)**: Strong evidence of a shift
                        - **Positive change**: Precipitation increased after the change point
                        - **Negative change**: Precipitation decreased after the change point

                        **U Statistic:**
                        - Measures the magnitude of the change
                        - Higher values = stronger evidence of change
                        - Maximum absolute value indicates the most likely change point
                        """)

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")


def _render_flow_duration_and_spells():
    """Render Flow Duration Curve and Wet/Dry Spell analysis"""
    st.markdown("### 💧 Flow Duration & Wet/Dry Spell Analysis")

    analyzer = st.session_state.hydro_analyzer

    # Create two sub-sections
    analysis_type = st.radio(
        "Select analysis type:",
        options=['Flow Duration Curve', 'Wet/Dry Spell Analysis'],
        horizontal=True
    )

    if analysis_type == 'Flow Duration Curve':
        st.info("""
        **What is a Flow Duration Curve (FDC)?**
        Shows the percentage of time that precipitation exceeds a given value.
        Essential for understanding variability and identifying high/low precipitation regimes.
        """)

        if st.button("📊 Calculate Flow Duration Curve", type="primary"):
            with st.spinner("🔄 Calculating Flow Duration Curve..."):
                try:
                    fdc_results = analyzer.calculate_flow_duration_curve()

                    if 'error' in fdc_results:
                        st.error(f"❌ {fdc_results['error']}")
                    else:
                        # Key percentiles
                        st.markdown("#### 📊 Key Percentiles")
                        percentiles = fdc_results.get('percentiles', {})

                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Q10 (High Flow)", f"{percentiles.get('Q10', 0):.1f} mm")
                        with col2:
                            st.metric("Q50 (Median)", f"{percentiles.get('Q50', 0):.1f} mm")
                        with col3:
                            st.metric("Q90 (Low Flow)", f"{percentiles.get('Q90', 0):.1f} mm")
                        with col4:
                            wet_days = fdc_results.get('num_wet_days', 0)
                            total_days = fdc_results.get('total_days', 1)
                            wet_pct = (wet_days / total_days) * 100
                            st.metric("Wet Days", f"{wet_pct:.1f}%")

                        # Visualization
                        fdc_plot = analyzer.create_flow_duration_curve_plot(fdc_results)
                        st.plotly_chart(fdc_plot, use_container_width=True)

                        with st.expander("📚 Understanding Flow Duration Curves"):
                            st.markdown("""
                            **How to Read FDC:**
                            - **X-axis**: Percentage of time a value is exceeded
                            - **Y-axis**: Precipitation value (mm/day)
                            - **Steep curve**: High variability
                            - **Flat curve**: Consistent precipitation

                            **Key Points:**
                            - **Q10**: Exceeded 10% of time (high precipitation regime)
                            - **Q50**: Exceeded 50% of time (median value)
                            - **Q90**: Exceeded 90% of time (low precipitation regime)
                            """)

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

    else:  # Wet/Dry Spell Analysis
        st.info("""
        **What is Wet/Dry Spell Analysis?**
        Analyzes consecutive periods of wet and dry days.
        Important for drought assessment and agricultural planning.
        """)

        # Threshold selection
        wet_threshold = st.slider(
            "Wet day threshold (mm):",
            min_value=0.1,
            max_value=10.0,
            value=1.0,
            step=0.1,
            help="Days with precipitation >= this value are considered 'wet'"
        )

        if st.button("📊 Analyze Wet/Dry Spells", type="primary"):
            with st.spinner("🔄 Analyzing wet and dry spells..."):
                try:
                    spell_results = analyzer.calculate_wet_dry_spell_analysis(wet_threshold=wet_threshold)

                    if 'error' in spell_results:
                        st.error(f"❌ {spell_results['error']}")
                    else:
                        # Summary metrics
                        st.markdown("#### 📊 Spell Statistics Summary")

                        col1, col2 = st.columns(2)

                        with col1:
                            st.markdown("**☀️ Dry Spells**")
                            dry_stats = spell_results.get('dry_spells', {})
                            st.metric("Maximum Length", f"{dry_stats.get('max_length', 0)} days")
                            st.metric("Average Length", f"{dry_stats.get('mean_length', 0):.1f} days")
                            st.metric("Number of Spells", dry_stats.get('num_spells', 0))

                        with col2:
                            st.markdown("**🌧️ Wet Spells**")
                            wet_stats = spell_results.get('wet_spells', {})
                            st.metric("Maximum Length", f"{wet_stats.get('max_length', 0)} days")
                            st.metric("Average Length", f"{wet_stats.get('mean_length', 0):.1f} days")
                            st.metric("Number of Spells", wet_stats.get('num_spells', 0))

                        # Drought risk
                        st.markdown("#### ⚠️ Drought Risk Assessment")
                        drought_risk = spell_results.get('drought_risk', {})

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            prob_7 = drought_risk.get('probability_dry_spell_gt_7days', 0) * 100
                            st.metric("Dry Spell > 7 days", f"{prob_7:.1f}%")
                        with col2:
                            prob_14 = drought_risk.get('probability_dry_spell_gt_14days', 0) * 100
                            st.metric("Dry Spell > 14 days", f"{prob_14:.1f}%")
                        with col3:
                            prob_30 = drought_risk.get('probability_dry_spell_gt_30days', 0) * 100
                            st.metric("Dry Spell > 30 days", f"{prob_30:.1f}%")

                        # Visualization
                        spell_plot = analyzer.create_wet_dry_spell_plot(spell_results)
                        st.plotly_chart(spell_plot, use_container_width=True)

                        with st.expander("📚 Understanding Spell Analysis"):
                            st.markdown("""
                            **Applications:**
                            - **Agriculture**: Identify drought-prone periods
                            - **Water Resources**: Assess water availability reliability
                            - **Irrigation Planning**: Determine irrigation frequency needs
                            - **Drought Monitoring**: Track dry period severity

                            **Interpreting Results:**
                            - **Long dry spells**: High drought risk
                            - **Short wet spells**: Irregular rainfall pattern
                            - **High probability of 30+ day dry spells**: Significant drought concern
                            """)

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")


def _reset_hydrology_configuration():
    """Reset hydrology configuration"""
    keys_to_reset = [
        'hydro_geometry', 'hydro_area_km2', 'hydro_dataset_info',
        'hydro_start_date', 'hydro_end_date', 'hydro_precipitation_data',
        'hydro_run_analysis', 'hydro_show_sample', 'hydro_analyzer'
    ]

    for key in keys_to_reset:
        if key in st.session_state:
            del st.session_state[key]
