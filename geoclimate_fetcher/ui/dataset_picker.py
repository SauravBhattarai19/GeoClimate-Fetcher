"""
Dataset picker widget for selecting Google Earth Engine datasets.
"""
import ipywidgets as widgets
from IPython.display import display, clear_output
from typing import Optional, Callable, Dict, Any, List
import pandas as pd

from geoclimate_fetcher.core.metadata import MetadataCatalog

class DatasetPickerWidget:
    """Widget for searching and selecting Google Earth Engine datasets."""
    
    def __init__(self, metadata_catalog: MetadataCatalog, 
                on_dataset_selected: Optional[Callable] = None):
        """
        Initialize the dataset picker widget.
        
        Args:
            metadata_catalog: MetadataCatalog instance
            on_dataset_selected: Callback function to execute after dataset selection
        """
        self.metadata_catalog = metadata_catalog
        self.on_dataset_selected = on_dataset_selected
        self.selected_dataset = None
        
        # Create UI components
        self.title = widgets.HTML("<h3>Dataset Selection</h3>")
        
        self.category_dropdown = widgets.Dropdown(
            options=[('All', 'all')] + [(cat, cat) for cat in self.metadata_catalog.categories],
            description='Category:',
            style={'description_width': 'initial'}
        )
        
        self.search_input = widgets.Text(
            value='',
            placeholder='Search datasets',
            description='Search:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='400px')
        )
        
        self.search_button = widgets.Button(
            description='Search',
            button_style='info',
            icon='search'
        )
        
        self.dataset_dropdown = widgets.Dropdown(
            options=[],
            description='Dataset:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='500px')
        )
        
        self.dataset_info = widgets.HTML(value='')
        
        self.select_button = widgets.Button(
            description='Select Dataset',
            button_style='success',
            icon='check',
            disabled=True
        )
        
        self.output = widgets.Output()
        
        # Bind events
        self.search_button.on_click(self._on_search_button_click)
        self.search_input.on_submit(lambda sender: self._on_search_button_click(None))
        self.category_dropdown.observe(self._on_category_change, names='value')
        self.dataset_dropdown.observe(self._on_dataset_change, names='value')
        self.select_button.on_click(self._on_select_button_click)
        
        # Layout
        search_box = widgets.HBox([self.search_input, self.search_button])
        
        # Main widget
        self.widget = widgets.VBox([
            self.title,
            widgets.HBox([
                self.category_dropdown,
                search_box
            ]),
            self.dataset_dropdown,
            self.dataset_info,
            self.select_button,
            self.output
        ])
        
        # Initial load of datasets
        self._load_datasets()
        
    def display(self):
        """Display the dataset picker widget."""
        display(self.widget)
        
    def _load_datasets(self, category: str = 'all', search_query: str = ''):
        """Load datasets based on category and search query."""
        try:
            if category == 'all' and not search_query:
                # All datasets
                df = self.metadata_catalog.all_datasets
            elif category != 'all' and not search_query:
                # Filter by category
                df = self.metadata_catalog.get_datasets_by_category(category)
            elif search_query:
                # Search across all or within category
                df = self.metadata_catalog.search_datasets(search_query)
                if category != 'all':
                    # Further filter by category
                    category_df = self.metadata_catalog.get_datasets_by_category(category)
                    df = pd.merge(df, category_df, how='inner')
            else:
                df = pd.DataFrame()
                
            # Update dropdown options
            if df.empty:
                self.dataset_dropdown.options = []
                self.dataset_info.value = '<p>No datasets found.</p>'
                self.select_button.disabled = True
            else:
                # Get dataset names
                dataset_names = df['Dataset Name'].tolist()
                self.dataset_dropdown.options = [(name, name) for name in dataset_names]
                
                # Select the first dataset by default
                if dataset_names:
                    self.dataset_dropdown.value = dataset_names[0]
                    
        except Exception as e:
            with self.output:
                clear_output()
                print(f"Error loading datasets: {str(e)}")
                
    def _on_category_change(self, change):
        """Handle category selection change."""
        if change.new:
            self._load_datasets(category=change.new, search_query=self.search_input.value)
            
    def _on_search_button_click(self, button):
        """Handle search button click."""
        with self.output:
            clear_output()
            search_query = self.search_input.value.strip()
            category = self.category_dropdown.value
            self._load_datasets(category=category, search_query=search_query)
            
    def _on_dataset_change(self, change):
        """Handle dataset selection change."""
        if change.new:
            # Get dataset info
            dataset = self.metadata_catalog.get_dataset_by_name(change.new)
            
            if dataset is not None:
                # Enable select button
                self.select_button.disabled = False
                
                # Format dataset info
                info_html = f"""
                <table style='width:100%; border-collapse: collapse;'>
                    <tr>
                        <td style='padding:5px; width:150px;'><b>Earth Engine ID:</b></td>
                        <td style='padding:5px;'>{dataset.get('Earth Engine ID', '')}</td>
                    </tr>
                    <tr>
                        <td style='padding:5px;'><b>Provider:</b></td>
                        <td style='padding:5px;'>{dataset.get('Provider', '')}</td>
                    </tr>
                    <tr>
                        <td style='padding:5px;'><b>Pixel Size:</b></td>
                        <td style='padding:5px;'>{dataset.get('Pixel Size (m)', '')} m</td>
                    </tr>
                    <tr>
                        <td style='padding:5px;'><b>Time Range:</b></td>
                        <td style='padding:5px;'>{dataset.get('Start Date', '')} to {dataset.get('End Date', '')}</td>
                    </tr>
                    <tr>
                        <td style='padding:5px;'><b>Temporal Resolution:</b></td>
                        <td style='padding:5px;'>{dataset.get('Temporal Resolution', '')}</td>
                    </tr>
                    <tr>
                        <td style='padding:5px;'><b>Band Names:</b></td>
                        <td style='padding:5px;'>{dataset.get('Band Names', '')}</td>
                    </tr>
                    <tr>
                        <td style='padding:5px;'><b>Description:</b></td>
                        <td style='padding:5px;'>{dataset.get('Description', '')}</td>
                    </tr>
                </table>
                """
                
                self.dataset_info.value = info_html
            else:
                self.dataset_info.value = '<p>Dataset information not available.</p>'
                self.select_button.disabled = True
                
    def _on_select_button_click(self, button):
        """Handle dataset selection button click."""
        dataset_name = self.dataset_dropdown.value
        
        if dataset_name:
            self.selected_dataset = self.metadata_catalog.get_dataset_by_name(dataset_name)
            
            with self.output:
                clear_output()
                print(f"Selected dataset: {dataset_name}")
                
                if self.on_dataset_selected:
                    self.on_dataset_selected(self.selected_dataset)
                    
    def get_selected_dataset(self) -> Optional[pd.Series]:
        """
        Get the currently selected dataset.
        
        Returns:
            Selected dataset as a Pandas Series or None if none selected
        """
        return self.selected_dataset