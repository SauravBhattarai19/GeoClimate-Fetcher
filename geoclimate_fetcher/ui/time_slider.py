"""
Time slider widget for selecting temporal range.
"""
import ipywidgets as widgets
from IPython.display import display, clear_output
from typing import Optional, Callable, Dict, Any, List, Tuple, Union
import pandas as pd
from datetime import datetime, date

from geoclimate_fetcher.core.metadata import MetadataCatalog

class TimeSliderWidget:
    """Widget for selecting temporal range for Earth Engine datasets."""
    
    def __init__(self, metadata_catalog: MetadataCatalog, 
                on_dates_selected: Optional[Callable] = None):
        """
        Initialize the time slider widget.
        
        Args:
            metadata_catalog: MetadataCatalog instance
            on_dates_selected: Callback function to execute after date selection
        """
        self.metadata_catalog = metadata_catalog
        self.on_dates_selected = on_dates_selected
        self.dataset_name = None
        self.start_date = None
        self.end_date = None
        
        # Create UI components
        self.title = widgets.HTML("<h3>Time Range Selection</h3>")
        
        self.date_range_text = widgets.HTML(value="<p>Dataset date range: N/A</p>")
        
        self.start_date_picker = widgets.DatePicker(
            description='Start date:',
            disabled=False,
            style={'description_width': 'initial'}
        )
        
        self.end_date_picker = widgets.DatePicker(
            description='End date:',
            disabled=False,
            style={'description_width': 'initial'}
        )
        
        self.apply_button = widgets.Button(
            description='Apply Date Range',
            button_style='success',
            icon='check',
            disabled=True
        )
        
        self.output = widgets.Output()
        
        # Bind events
        self.apply_button.on_click(self._on_apply_button_click)
        
        # Main widget
        self.widget = widgets.VBox([
            self.title,
            self.date_range_text,
            widgets.HBox([self.start_date_picker, self.end_date_picker]),
            self.apply_button,
            self.output
        ])
        
    def display(self):
        """Display the time slider widget."""
        display(self.widget)
        
    def set_dataset(self, dataset_or_name: Union[pd.Series, str]):
        """
        Set the current dataset and update available date range.
        
        Args:
            dataset_or_name: Dataset as a Pandas Series or dataset name
        """
        if isinstance(dataset_or_name, pd.Series):
            dataset_name = dataset_or_name.get('Dataset Name')
        else:
            dataset_name = dataset_or_name
            
        self.dataset_name = dataset_name
        
        # Get date range for this dataset
        start_date_str, end_date_str = self.metadata_catalog.get_date_range(dataset_name)
        
        # Update UI
        if start_date_str and end_date_str:
            # Parse dates
            try:
                with self.output:
                    clear_output()
                    print(f"Parsing date range: {start_date_str} to {end_date_str}")
                
                start_date = self._parse_date(start_date_str)
                end_date = self._parse_date(end_date_str)
                
                # Update date pickers
                self.start_date_picker.value = start_date
                self.end_date_picker.value = end_date
                
                # Update date range text
                self.date_range_text.value = f"<p>Dataset date range: {start_date_str} to {end_date_str}</p>"
                
                # Enable apply button
                self.apply_button.disabled = False
                
            except Exception as e:
                with self.output:
                    clear_output()
                    print(f"Error parsing date range: {str(e)}")
                    print(f"Attempting to use alternative date format for {start_date_str} and {end_date_str}")
                    
                    # Try to parse with a set of fallback formats and defaults
                    try:
                        # Try to parse with a more forgiving approach
                        start_date = self._safe_parse_date(start_date_str)
                        end_date = self._safe_parse_date(end_date_str)
                        
                        print(f"Successfully parsed using alternative methods: {start_date} to {end_date}")
                        
                        # Update date pickers
                        self.start_date_picker.value = start_date
                        self.end_date_picker.value = end_date
                        
                        # Update date range text
                        self.date_range_text.value = f"<p>Dataset date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}</p>"
                        
                        # Enable apply button
                        self.apply_button.disabled = False
                    except Exception as e2:
                        print(f"All parsing attempts failed: {str(e2)}")
                        
                        # Fallback to current date range
                        today = date.today()
                        one_month_ago = date(today.year, today.month-1 if today.month > 1 else 12, 1)
                        
                        self.start_date_picker.value = one_month_ago
                        self.end_date_picker.value = today
                        
                        self.date_range_text.value = f"<p>Unable to parse dataset dates. Using default range.</p>"
                        self.apply_button.disabled = False
        else:
            self.date_range_text.value = "<p>Dataset date range: Not available</p>"
            
            # Fallback to a reasonable default
            today = date.today()
            one_month_ago = date(today.year, today.month-1 if today.month > 1 else 12, 1)
            
            self.start_date_picker.value = one_month_ago
            self.end_date_picker.value = today
            
            self.apply_button.disabled = False
            
    def _safe_parse_date(self, date_str: str) -> date:
        """
        A more robust date parser that tries multiple formats and provides sensible defaults.
        """
        # Try all common formats
        formats = [
            '%Y-%m-%d', '%Y/%m/%d', '%Y%m%d',  # ISO-like formats
            '%m/%d/%Y', '%m-%d-%Y', '%m.%d.%Y',  # US formats
            '%d/%m/%Y', '%d-%m-%Y', '%d.%m.%Y',  # European formats
            '%b %d, %Y', '%B %d, %Y',  # Month name formats
            '%Y-%m', '%Y/%m', '%m/%Y', '%m-%Y',  # Year-month formats
            '%Y'  # Just year
        ]
        
        # First try with standard formats
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt).date()
            except ValueError:
                continue
                
        # Handle special cases
        if '/' in date_str and len(date_str.split('/')) == 3:
            # Try to intelligently parse m/d/y format
            parts = date_str.split('/')
            if len(parts[2]) == 4:  # Assume year is the 4-digit part
                try:
                    month, day, year = int(parts[0]), int(parts[1]), int(parts[2])
                    return date(year, month, day)
                except:
                    pass
        
        # If nothing works, use current date
        print(f"Could not parse date '{date_str}', using today's date")
        return date.today()
            
    def _parse_date(self, date_str: str) -> date:
        """Parse a date string in various formats."""
        for fmt in [
            '%Y-%m-%d', '%Y/%m/%d', '%Y%m%d',  # ISO-like formats
            '%m/%d/%Y', '%m-%d-%Y',  # US formats
            '%d/%m/%Y', '%d-%m-%Y',  # European formats
            '%Y-%m', '%Y/%m',  # Year-month formats
            '%Y'  # Just year
        ]:
            try:
                return datetime.strptime(date_str, fmt).date()
            except ValueError:
                continue
                
        raise ValueError(f"Unable to parse date: {date_str}")
        
    def _on_apply_button_click(self, button):
        """Handle apply button click."""
        start_date = self.start_date_picker.value
        end_date = self.end_date_picker.value
        
        if start_date and end_date:
            if start_date > end_date:
                with self.output:
                    clear_output()
                    print("Error: Start date must be before end date.")
                return
                
            self.start_date = start_date
            self.end_date = end_date
            
            with self.output:
                clear_output()
                print(f"Selected date range: {start_date} to {end_date}")
                
                if self.on_dates_selected:
                    self.on_dates_selected(start_date, end_date)
        else:
            with self.output:
                clear_output()
                print("Please select both start and end dates.")
                
    def get_selected_dates(self) -> Tuple[Optional[date], Optional[date]]:
        """
        Get the currently selected date range.
        
        Returns:
            Tuple of (start_date, end_date)
        """
        return self.start_date, self.end_date