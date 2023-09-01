#!/usr/bin/env python
# coding: utf-8

from bim.gis import osm
from bim.gis.traffic import od_matrix
from bim.gis.traffic import analytics
from bim.gis.lsoa import LSOABoundaries
from bim.gis.lsoa import CountiesBoundary

area_od_matrix = od_matrix.AreaOriginDistanceMatrix()
od_gm_list = area_od_matrix.load()
# test od matrix analytics

od_matrix_analysis = analytics.ODMatrixAnalysis()
od_matrix_analysis.compare_population_and_od_commuter_numbers('bus')
od_matrix_analysis.compare_population_and_od_commuter_numbers()

lsoa_boundaries = LSOABoundaries()
lsoa_boundaries_gdf = lsoa_boundaries.load()
streets = osm.Nominatim().load_streets()
counties_regions = CountiesBoundary()
counties_region_gdf = counties_regions.load()
gm_boundary = counties_regions.get_gm_boundary()
gm_boundary.plot()

gm_boundary_poly = counties_regions.get_gm_boundary_polygon()
gm_lsoa_boundaries_gdf = area_od_matrix.get_gm_lsoa_boundaries(counties_regions, lsoa_boundaries)

area_od_matrix.plot_all_connected_lsoa()
area_od_matrix.plot_top_manchester_routes(3)
