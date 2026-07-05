"""
Extract and cache table metadata (column headers, metric names) from original GRI tables.
"""

import csv
import ast
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class ColumnMetadata:
    """Metadata about a single table column"""
    col_idx: int
    col_name: str
    is_year: bool
    year: Optional[str] = None
    is_metric: bool = False


@dataclass
class MetricRowMetadata:
    """Metadata about a data row (metric)"""
    row_idx: int
    metric_name: str
    metric_type: str
    unit: str


@dataclass
class TableMetadata:
    """Complete metadata for a single table"""
    table_id: str
    columns: List[ColumnMetadata]
    metrics: List[MetricRowMetadata]
    header_row: List[str]
    csv_path: str


class TableMetadataExtractor:
    """Extract metadata from GRI annotation tables"""
    
    METRIC_PATTERNS = {
        'waste': ['waste', 'disposed', 'landfill', 'recycl'],
        'energy': ['energy', 'consumption', 'gwh', 'mwh', 'power', 'electricity'],
        'emissions': ['emissions', 'co2', 'ghg', 'carbon', 'scope', 'greenhouse'],
        'water': ['water', 'withdrawn', 'discharge', 'm3'],
        'emissions_intensity': ['intensity', 'per employee', 'per unit', 'per revenue'],
    }
    
    def __init__(self, annotation_root: Path):
        self.annotation_root = Path(annotation_root)
        self.cache: Dict[str, TableMetadata] = {}
    
    @staticmethod
    def extract_metric_type(metric_name: str) -> str:
        """Extract metric category from metric row name"""
        metric_lower = metric_name.lower()
        
        for category, patterns in TableMetadataExtractor.METRIC_PATTERNS.items():
            for pattern in patterns:
                if pattern in metric_lower:
                    return category
        
        return 'other'
    
    @staticmethod
    def extract_unit(metric_name: str) -> str:
        """Extract unit from metric name"""
        import re
        match = re.search(r'\(([^)]+)\)', metric_name)
        if match:
            return match.group(1)
        return ''
    
    @staticmethod
    def extract_year_from_header(col_name: str) -> Optional[str]:
        """Extract year from column header if present"""
        import re
        match = re.search(r'\b(19|20)\d{2}\b', col_name)
        if match:
            return match.group(0)
        return None
    
    def load_table(self, pdf_name: str, page_nbr: int, table_nbr: int) -> Optional[TableMetadata]:
        """Load and parse a single table"""
        table_id = f"{pdf_name}_{page_nbr}_{table_nbr}"
        
        if table_id in self.cache:
            return self.cache[table_id]
        
        table_path = self.annotation_root / pdf_name / f"{page_nbr}_{table_nbr}.csv"
        
        if not table_path.exists():
            return None
        
        try:
            with open(table_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f, delimiter=';')
                rows = list(reader)
        except Exception as e:
            return None
        
        if not rows:
            return None
        
        header = rows[0]
        columns = self._parse_columns(header)
        metrics = self._parse_metrics(rows, columns)
        
        metadata = TableMetadata(
            table_id=table_id,
            columns=columns,
            metrics=metrics,
            header_row=header,
            csv_path=str(table_path)
        )
        
        self.cache[table_id] = metadata
        return metadata
    
    def _parse_columns(self, header: List[str]) -> List[ColumnMetadata]:
        """Parse column headers into metadata"""
        columns = []
        
        for col_idx, col_name in enumerate(header):
            col_name = col_name.strip()
            
            year = self.extract_year_from_header(col_name)
            is_year = year is not None
            is_metric = any(pattern in col_name.lower() 
                          for pattern in ['%', 'change', 'target', 'base year'])
            
            columns.append(ColumnMetadata(
                col_idx=col_idx,
                col_name=col_name,
                is_year=is_year,
                year=year,
                is_metric=is_metric
            ))
        
        return columns
    
    def _parse_metrics(self, rows: List[List[str]], 
                      columns: List[ColumnMetadata]) -> List[MetricRowMetadata]:
        """Parse metric rows into metadata"""
        metrics = []
        
        for row_idx, row in enumerate(rows[1:], start=1):
            if not row or not row[0].strip():
                continue
            
            metric_name = row[0].strip()
            metric_type = self.extract_metric_type(metric_name)
            unit = self.extract_unit(metric_name)
            
            metrics.append(MetricRowMetadata(
                row_idx=row_idx,
                metric_name=metric_name,
                metric_type=metric_type,
                unit=unit
            ))
        
        return metrics
    
    def get_value_at(self, pdf_name: str, page_nbr: int, table_nbr: int,
                    row_idx: int, col_idx: int) -> Optional[Tuple[str, Dict]]:
        """Get value at specific cell with context"""
        metadata = self.load_table(pdf_name, page_nbr, table_nbr)
        if not metadata:
            return None
        
        table_path = Path(metadata.csv_path)
        try:
            with open(table_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f, delimiter=';')
                rows = list(reader)
        except:
            return None
        
        if row_idx >= len(rows) or col_idx >= len(rows[row_idx]):
            return None
        
        value = rows[row_idx][col_idx].strip()
        
        metric = next((m for m in metadata.metrics if m.row_idx == row_idx), None)
        column = metadata.columns[col_idx] if col_idx < len(metadata.columns) else None
        
        context = {
            'table_id': metadata.table_id,
            'value': value,
            'row_idx': row_idx,
            'col_idx': col_idx,
        }
        
        if metric:
            context['metric_name'] = metric.metric_name
            context['metric_type'] = metric.metric_type
            context['unit'] = metric.unit
        
        if column:
            context['col_name'] = column.col_name
            context['is_year'] = column.is_year
            context['year'] = column.year
        
        return value, context
