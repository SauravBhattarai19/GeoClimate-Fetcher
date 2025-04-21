"""
Band picker widget for selecting bands from a dataset.
"""
import ipywidgets as widgets
from IPython.display import display, clear_output
from typing import Optional, Callable, Dict, Any, List
import pandas as pd
from typing import Optional, Callable, Dict, Any, List, Union

from geoclimate_fetcher.core.metadata import MetadataCatalog

class BandPickerWidget:
    """Widget for selecting bands from a Google Earth Engine dataset."""
    
    def __init__(self, metadata_catalog: MetadataCatalog, 
                on_bands_selected: Optional[Callable] = None):
        """
        Initialize the band picker widget.
        
        Args:
            metadata_catalog: MetadataCatalog instance
            on_bands_selected: Callback function to execute after band selection
        """
        self.metadata_catalog = metadata_catalog
        self.on_bands_selected = on_bands_selected
        self.dataset_name = None
        self.selected_bands = []
        
        # Create UI components
        self.title = widgets.HTML("<h3>Band Selection</h3>")
        
        self.band_checkboxes = widgets.VBox([])
        
        self.select_all_button = widgets.Button(
            description='Select All',
            button_style='info',
            icon='check-square'
        )
        
        self.clear_all_button = widgets.Button(
            description='Clear All',
            button_style='warning',
            icon='square'
        )
        
        self.apply_button = widgets.Button(
            description='Apply Selection',
            button_style='success',
            icon='check',
            disabled=True
        )
        
        self.output = widgets.Output()
        
        # Bind events
        self.select_all_button.on_click(self._on_select_all_button_click)
        self.clear_all_button.on_click(self._on_clear_all_button_click)
        self.apply_button.on_click(self._on_apply_button_click)
        
        # Layout
        buttons = widgets.HBox([
            self.select_all_button,
            self.clear_all_button,
            self.apply_button
        ])
        
        # Main widget
        self.widget = widgets.VBox([
            self.title,
            widgets.Label("Select bands to include:"),
            self.band_checkboxes,
            buttons,
            self.output
        ])
        
    def display(self):
        """Display the band picker widget."""
        display(self.widget)
        
    def set_dataset(self, dataset_or_name: Union[pd.Series, str]):
        """
        Set the current dataset and update available bands.
        
        Args:
            dataset_or_name: Dataset as a Pandas Series or dataset name
        """
        if isinstance(dataset_or_name, pd.Series):
            dataset_name = dataset_or_name.get('Dataset Name')
        else:
            dataset_name = dataset_or_name
            
        self.dataset_name = dataset_name
        
        # Get bands for this dataset
        bands = self.metadata_catalog.get_bands_for_dataset(dataset_name)
        
        # Update checkboxes
        self._update_band_checkboxes(bands)
        
    def _update_band_checkboxes(self, bands: List[str]):
        """Update the band checkboxes based on available bands."""
        # Clear existing checkboxes
        self.band_checkboxes.children = []
        
        if not bands:
            with self.output:
                clear_output()
                print("No bands available for this dataset.")
            return
            
        # Create a checkbox for each band
        checkboxes = []
        for band in bands:
            checkbox = widgets.Checkbox(
                value=False,
                description=band,
                indent=False
            )
            checkboxes.append(checkbox)
            
        # Update the widget
        self.band_checkboxes.children = checkboxes
        
        # Enable apply button if bands are available
        self.apply_button.disabled = len(bands) == 0
        
    def _on_select_all_button_click(self, button):
        """Handle select all button click."""
        for checkbox in self.band_checkboxes.children:
            checkbox.value = True
            
    def _on_clear_all_button_click(self, button):
        """Handle clear all button click."""
        for checkbox in self.band_checkboxes.children:
            checkbox.value = False
            
    def _on_apply_button_click(self, button):
        """Handle apply button click."""
        # Get selected bands
        selected = []
        for checkbox in self.band_checkboxes.children:
            if checkbox.value:
                selected.append(checkbox.description)
                
        self.selected_bands = selected
        
        with self.output:
            clear_output()
            
            if selected:
                print(f"Selected bands: {', '.join(selected)}")
                
                if self.on_bands_selected:
                    self.on_bands_selected(selected)
            else:
                print("No bands selected. Please select at least one band.")
                
    def get_selected_bands(self) -> List[str]:
        """
        Get the currently selected bands.
        
        Returns:
            List of selected band names
        """
        return self.selected_bands