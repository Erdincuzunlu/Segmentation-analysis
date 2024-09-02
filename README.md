# Segmentation-analysis

# Persona Analysis and Customer Segmentation

This project involves analyzing a dataset from a game company to define new level-based customer personas and segment them based on their potential revenue contribution.

## Project Structure

- **Data Loading**: The dataset is loaded and initial exploratory data analysis is performed.
- **Feature Analysis**: Unique values and sales per country are computed.
- **Revenue Analysis**: Total and average revenues are calculated based on different categorical breakdowns (e.g., country, source).
- **Customer Segmentation**: The data is used to define customer personas based on country, source, gender, and age. These personas are then segmented based on their average revenue.
- **New Customer Classification**: The model is used to predict the potential revenue from new customers.

## Usage

1. Load the dataset from the specified path.
2. Run the script to perform the analysis and segmentation.
3. Review the results to understand which customer segments are most valuable and predict future customer revenue.

## Key Functions

- **all_describe**: Provides a comprehensive summary of the DataFrame.
- **agg_df**: Aggregates data by `COUNTRY`, `SOURCE`, `SEX`, and `AGE`.
- **customers_level_based**: Creates new customer personas by combining various categorical features.
- **Segment Analysis**: Segments customers into different groups based on their potential revenue.

## Example

Two examples of how to use the persona model to predict revenue for new customers are included in the script.
