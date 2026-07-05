## Loading the GRI-QA Dataset

### Dataset Structure

The GRI-QA dataset is organized into two categories:

- **`one-table/`** - Questions requiring single-table reasoning
- **`multi-table/`** - Questions requiring multi-table joins and reasoning

### Table Reference System

Each dataset sample includes table metadata in three columns:

| Column | Description | Example |
|--------|-------------|---------|
| `pdf_name` | Source document name (with `.pdf` extension) | `axa_2023.pdf` |
| `page_nbr` | Page number within the document | `42` |
| `table_nbr` | Table number on that page | `3` |

### Accessing Table Data

To retrieve the actual table data for a sample:

1. **Remove the `.pdf` extension** from the `pdf_name` value
2. **Construct the path** to the CSV file:
