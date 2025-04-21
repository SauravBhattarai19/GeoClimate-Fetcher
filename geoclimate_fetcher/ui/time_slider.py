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
                self.apply_button.disabled = True
        else:
            self.date_range_text.value = "<p>Dataset date range: Not available</p>"
            self.apply_button.disabled = True
            
    def _parse_date(self, date_str: str) -> date:
        """Parse a date string in various formats."""
        for fmt in ['%Y-%m-%d', '%Y/%m/%d', '%Y%m%d', '%Y-%m', '%Y/%m', '%Y']:
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