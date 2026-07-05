# Agentic Top-50 Numeric Outlier Slice

## Mode Counts
- M5_same_table_wrong_column_or_row_passthrough: 25/50
- M2_percent_ratio_scale_leak_absolute_value_returned: 9/50
- M1_zero_gold_denominator_with_nonzero_prediction: 7/50
- M3_wrong_table_selected: 5/50
- M6_other_numeric_selection: 3/50
- M4_operation_applied_to_misaligned_operands: 1/50

## Representative Cases

### M5_same_table_wrong_column_or_row_passthrough
- single_table_extractive-1770 | nre=1227047.46 | gold=5.2 | pred=6380652 | top_primary=6380652 | expected_table=0 observed_table=0 | source=initial calc=False
  question: What percentage of CO2e emissions was attributed to rail transport in 2023?
- multi_table_multistep-56 | nre=735810.38 | gold=6.72 | pred=4944652.5 | top_primary=4944652.5 | expected_table=0 observed_table=0 | source=structured_table_lookup calc=False
  question: What is the highest water consumption calculated as the average of the 2023 and 2022 consumption for the following companies in million cubic meters?
- single_table_extractive-1602 | nre=610384.48 | gold=10.4 - 10.1 | pred=6348009 | top_primary=6348009 | expected_table=0 observed_table=0 | source=initial calc=False
  question: What are the maximum and minimum fuel consumption (in l/100 km) for the BMW M3 CS?
- multi_table_multistep-126 | nre=484768.85 | gold=10.2 | pred=4944652.5 | top_primary=4944652.5 | expected_table=0 observed_table=0 | source=structured_table_lookup calc=False
  question: What is the highest water consumption calculated as the average of the 2023 and 2022 consumption for the following companies in millions of cu.m?
- single_table_extractive-1686 | nre=443626.05 | gold=14.6 | pred=6476955 | top_primary=6476955 | expected_table=0 observed_table=0 | source=initial calc=False
  question: What was the percentage of groundwater in total water consumption in 2021?

### M2_percent_ratio_scale_leak_absolute_value_returned
- single_table_extractive-1693 | nre=63806519.00 | gold=0.1 | pred=6380652 | top_primary=6380652 | expected_table=0 observed_table=0 | source=initial calc=False
  question: What is the percentage of surface water in total water consumption for the year 2023?
- single_table_extractive-1695 | nre=60408239.00 | gold=0.1 | pred=6040824 | top_primary=6040824 | expected_table=0 observed_table=0 | source=initial calc=False
  question: How much of the total water consumption was rainwater in percentage for the year 2020?
- single_table_extractive-1698 | nre=31903259.00 | gold=0.2 | pred=6380652 | top_primary=6380652 | expected_table=0 observed_table=0 | source=initial calc=False
  question: What is the percentage of rainwater in total water consumption for the year 2023?
- single_table_extractive-1696 | nre=21589849.00 | gold=0.3 | pred=6476955 | top_primary=6476955 | expected_table=0 observed_table=0 | source=initial calc=False
  question: What was the percentage of rainwater in total water consumption in 2021?
- single_table_extractive-1697 | nre=20986632.33 | gold=0.3 | pred=6295990 | top_primary=6295990 | expected_table=0 observed_table=0 | source=initial calc=False
  question: What percentage of the total water consumption was rainwater in 2022?

### M1_zero_gold_denominator_with_nonzero_prediction
- single_table_extractive-1691 | nre=6476955.00 | gold=0 | pred=6476955 | top_primary=6476955 | expected_table=0 observed_table=0 | source=initial calc=False
  question: What was the percentage of surface water in total water consumption in 2021?
- single_table_extractive-1692 | nre=6295990.00 | gold=0 | pred=6295990 | top_primary=6295990 | expected_table=0 observed_table=0 | source=initial calc=False
  question: What percentage of the total water consumption was surface water in 2022?
- single_table_extractive-1694 | nre=5226227.00 | gold=0 | pred=5226227 | top_primary=5226227 | expected_table=0 observed_table=0 | source=initial calc=False
  question: What percentage of the total water consumption was rainwater in 2019?
- single_table_extractive-1689 | nre=5226227.00 | gold=0 | pred=5226227 | top_primary=5226227 | expected_table=0 observed_table=0 | source=initial calc=False
  question: What percentage of the total water consumption was surface water in 2019?
- single_table_extractive-178 | nre=5049144.00 | gold=0 | pred=5049144 | top_primary=5049144 | expected_table=0 observed_table=0 | source=initial calc=False
  question: What is the total weight of high-level radioactive waste generated in 2023 in metric tons?

### M3_wrong_table_selected
- multi_table_multistep-4 | nre=564613.00 | gold=prosiebensat1-media | pred=564614 | top_primary=564614 | expected_table=1 observed_table=0 | source=calculation calc=True
  question: Which company has the lowest total energy consumption over the last 2 years among the following companies in GWh?
- multi_table_multistep-162 | nre=564613.00 | gold=munich-re, prosiebensat1-media | pred=564614 | top_primary=564614 | expected_table=1 observed_table=0 | source=calculation calc=True
  question: Which companies have the top 2 highest values of average energy consumed in the last 2 years (sorted in descending order) among the following companies in GWh?
- multi_table_multistep-149 | nre=564613.00 | gold=prosiebensat1-media, OTC_SU | pred=564614 | top_primary=564614 | expected_table=1 observed_table=0 | source=calculation calc=True
  question: Which companies have the top 2 highest values of average energy consumed in the last 2 years (sorted in ascending order) among the following companies in MWh?
- single_table_extractive-691 | nre=191484.80 | gold=5 | pred=957429 | top_primary=957429 | expected_table=1 observed_table=0 | source=initial calc=False
  question: What was the value of Processing of sold products in 2023 in Mt CO2e?
- single_table_extractive-1063 | nre=73202.70 | gold=1 | pred=73203.7 | top_primary=73203.7 | expected_table=1 observed_table=0 | source=initial calc=False
  question: How many disputes with suppliers were settled in 2021?

### M6_other_numeric_selection
- multi_table_quantitative-36 | nre=374152.27 | gold=20.38 | pred=7625243.59 | top_primary=28632509.0 | expected_table=0 observed_table=0 | source=multi_table_structured_join calc=False
  question: What is the average of total water consumption in the years 2023 and 2022 among the following companies in millions of m3?
- multi_table_quantitative-55 | nre=324339.43 | gold=23.51 | pred=7625243.59 | top_primary=28632509.0 | expected_table=0 observed_table=0 | source=multi_table_structured_join calc=False
  question: What is the average of total water consumption in the years 2023 and 2022 among the following companies in millions of m3?
- multi_table_quantitative-2 | nre=119147.12 | gold=1.11 | pred=132254.41 | top_primary=0.05 | expected_table=0 observed_table=0 | source=multi_table_structured_join calc=False
  question: What is the sum of GHG emissions Scope 2 in 2023 among the following companies in Million metric tons of CO2 equivalents?

### M4_operation_applied_to_misaligned_operands
- multi_table_multistep-48 | nre=97037.22 | gold=9.89, 80.6 | pred=959708 | top_primary=OTC_BAMGF | expected_table=0 observed_table=0 | source=calculation calc=True
  question: What are the top 2 lowest water consumption values (in ascending order) obtained by summing the 2023 and 2022 consumption for the following companies in millions of m3?