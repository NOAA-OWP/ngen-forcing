from swe_mapping.core.run_swe import swe_map

swe_map(['2015-12-01',
        'sample_data/sample_csv/13240000/',
        'sample_data/13240000.nc',
        'sample_data/sample_gpkg/gages-13240000.gpkg',
        'sample_data/simulated_map.png',
        'sample_data/raw_map.png',
        'sample_data/lumped_map.png'
        ])
