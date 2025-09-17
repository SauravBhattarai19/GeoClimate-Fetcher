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
    
    # Add home button
    if st.button("🏠 Back to Home"):
        st.session_state.app_mode = None
        st.rerun()
    
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
    
    # Results section - only show after user clicks Run Analysis
    if analysis_ready:
        if st.session_state.get('hydro_run_analysis', False):
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

                    # Enhanced post-download integration
                    st.markdown("---")
                    st.markdown("### 🎉 Data Download Complete!")

                    # Register the precipitation data
                    download_handler = get_download_handler("hydrology_analyzer")
                    result_id = register_csv_download(
                        "hydrology_analyzer",
                        f"precipitation_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        precipitation_data,
                        metadata={
                            'dataset': dataset_info.get('name', 'Unknown'),
                            'analysis_type': 'precipitation_download',
                            'records': len(precipitation_data),
                            'date_range': f"{start_date_str} to {end_date_str}",
                            'avg_precipitation': avg_precip
                        }
                    )

                    # Show quick visualization of the precipitation data
                    st.markdown("#### 📊 Quick Precipitation Analysis")
                    try:
                        # Create instant visualization
                        quick_visualizer.render_quick_csv_analysis(
                            precipitation_data,
                            title="Precipitation Time Series Analysis",
                            max_columns=3
                        )
                    except Exception as e:
                        st.error(f"Error creating quick visualization: {str(e)}")

                    # Show post-download options
                    render_post_download_integration("hydrology_analyzer", [result_id])

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

    # Analysis tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Time Series", "📊 Statistics", "🌧️ Return Periods", "📚 Educational", "💾 Downloads"])

    with tab1:
        _render_time_series_analysis()

    with tab2:
        _render_statistical_analysis()

    with tab3:
        _render_return_period_analysis()

    with tab4:
        _render_educational_content()

    with tab5:
        _render_hydrology_downloads()


def _render_time_series_analysis():
    """Render time series analysis"""
    st.markdown("### 📈 Precipitation Time Series")
    
    # Generate sample time series data
    precipitation_data = st.session_state.hydro_precipitation_data
    
    # Create time series plot
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=precipitation_data['date'],
        y=precipitation_data['precipitation'],
        mode='lines',
        name='Daily Precipitation',
        line=dict(color='blue', width=1)
    ))
    
    fig.update_layout(
        title="Daily Precipitation Time Series",
        xaxis_title="Date",
        yaxis_title="Precipitation (mm)",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Monthly aggregation
    st.markdown("### 📅 Monthly Analysis")
    
    monthly_data = _calculate_monthly_stats(precipitation_data)
    
    fig_monthly = go.Figure()
    fig_monthly.add_trace(go.Bar(
        x=monthly_data['month'],
        y=monthly_data['total_precip'],
        name='Monthly Total',
        marker_color='lightblue'
    ))
    
    fig_monthly.update_layout(
        title="Monthly Precipitation Totals",
        xaxis_title="Month",
        yaxis_title="Precipitation (mm)",
        height=400
    )
    
    st.plotly_chart(fig_monthly, use_container_width=True)


def _render_statistical_analysis():
    """Render statistical analysis"""
    st.markdown("### 📊 Statistical Summary")
    
    precipitation_data = st.session_state.hydro_precipitation_data
    precip_values = precipitation_data['precipitation']
    
    # Basic statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Mean", f"{precip_values.mean():.2f} mm")
        st.metric("Median", f"{precip_values.median():.2f} mm")
    
    with col2:
        st.metric("Maximum", f"{precip_values.max():.2f} mm")
        st.metric("Minimum", f"{precip_values.min():.2f} mm")
    
    with col3:
        st.metric("Std Dev", f"{precip_values.std():.2f} mm")
        st.metric("Total Days", f"{len(precip_values)}")
    
    with col4:
        wet_days = (precip_values > 1.0).sum()
        st.metric("Wet Days", f"{wet_days}")
        st.metric("Wet Day %", f"{wet_days/len(precip_values)*100:.1f}%")
    
    # Distribution plot
    st.markdown("### 📊 Precipitation Distribution")
    
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
        st.markdown("#### 📊 Statistical Analysis")

        try:
            # Create statistical summary
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


def _reset_hydrology_configuration():
    """Reset hydrology configuration"""
    keys_to_reset = [
        'hydro_geometry', 'hydro_area_km2', 'hydro_dataset_info',
        'hydro_start_date', 'hydro_end_date', 'hydro_precipitation_data',
        'hydro_run_analysis', 'hydro_show_sample'
    ]

    for key in keys_to_reset:
        if key in st.session_state:
            del st.session_state[key]
