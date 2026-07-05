# Staged PR Top-50 Before/After

## Stage Summary
- Baseline (old metric) mean NRE: 4862337.683801424
- Stage A (pipeline changes, old metric) mean NRE: 316922.8992850249
- Baseline recomputed (new metric) mean NRE: 1.9999812129275956
- Stage B (pipeline + new metric) mean NRE: 1.8983158967342137
- Stage A improved IDs: 11
- Stage A worse IDs: 0

## Top 15 Largest Stage A Improvements (old metric)
- single_table_extractive-1693: delta=63806507.1220, before=63806519.0000, after=11.8780, source_after=initial
- single_table_extractive-1695: delta=60407710.4380, before=60408239.0000, after=528.5620, source_after=initial
- single_table_extractive-1698: delta=31903253.5610, before=31903259.0000, after=5.4390, source_after=initial
- single_table_extractive-1694: delta=5226226.1767, before=5226227.0000, after=0.8233, source_after=initial
- single_table_extractive-1689: delta=5226226.1767, before=5226227.0000, after=0.8233, source_after=initial
- single_table_extractive-1690: delta=4946864.1811, before=4946865.0000, after=0.8189, source_after=initial
- multi_table_quantitative-1: delta=661271.0000, before=661271.0000, after=0.0000, source_after=gpt_multi_table_tool
- single_table_extractive-1791: delta=286179.4762, before=286180.3636, after=0.8874, source_after=initial
- single_table_extractive-1683: delta=76139.4473, before=76140.4320, after=0.9846, source_after=initial
- single_table_extractive-1680: delta=69995.9748, before=69996.9606, after=0.9859, source_after=initial
- single_table_extractive-1063: delta=66373.7000, before=73202.7000, after=6829.0000, source_after=initial
- single_table_extractive-178: delta=0.0000, before=5049144.0000, after=5049144.0000, source_after=initial
- single_table_extractive-1602: delta=0.0000, before=610384.4808, after=610384.4808, source_after=initial
- multi_table_multistep-4: delta=0.0000, before=564613.0000, after=564613.0000, source_after=calculation
- multi_table_multistep-162: delta=0.0000, before=564613.0000, after=564613.0000, source_after=calculation