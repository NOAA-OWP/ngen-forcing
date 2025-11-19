from swe_timeseries.swe_timeseries import swe_ts

swe_ts(['sample_data/sample_csv/09359500', 
	  'sample_data/sample_gpkg/gauge_09359500.gpkg',
	  '--plot_output', 'comb_plot.png',
	  '--csv_output','comb_table.csv'
	  ])
