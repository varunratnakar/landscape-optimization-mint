import os
from osgeo import gdal
from osgeo import ogr
import geopandas as gpd
import pandas as pd
import numpy as np
import rasterio as rs
import xarray as xr
import affine
from rasterio.features import shapes
from shapely.geometry import Polygon, MultiPolygon
import warnings
warnings.filterwarnings("ignore")

import rasterio.features
import shapely.geometry as sg
import shutil
import rioxarray
import sys

import s3fs

args = sys.argv
# print(args, len(args))
print('Parameters received=', len(args))

url = args[1]
key=args[2]
secret=args[3]
fs = s3fs.S3FileSystem(key=key,
                       secret=secret,
                       client_kwargs={'endpoint_url': url,
                                      'verify': False})
fs.read_timeout = 600

print('S3 bucket initialized')

data = pd.read_csv(args[4])
filenames = data['filename'].values
filenames = filenames[:5]   # Need to remove this contraint while running!!!
print('Total ignition files=', len(filenames))

def polygonize(da: xr.DataArray) -> gpd.GeoDataFrame:
    """
    Polygonize a 2D-DataArray into a GeoDataFrame of polygons.

    Parameters
    ----------
    da : xr.DataArray

    Returns
    -------
    polygonized : geopandas.GeoDataFrame
    """
    if da.dims != ("y", "x"):
        raise ValueError('Dimensions must be ("y", "x")')

    values = da.values
    transform = da.attrs.get("transform", None)
    if transform is None:
        raise ValueError("transform is required in da.attrs")
    transform = affine.Affine(*transform)
    shapes = rasterio.features.shapes(values, transform=transform)

    geometries = []
    colvalues = []
    for (geom, colval) in shapes:
        geometries.append(sg.Polygon(geom["coordinates"][0]))
        colvalues.append(colval)

    gdf = gpd.GeoDataFrame({"value": colvalues, "geometry": geometries})
    gdf.crs = da.attrs.get("crs")
    return gdf

def make_polygons(filename, in_path, out_path, prefix):
    """
    Function takes fire footprint raster, converts the intensity values to binary values (burn/not burn) and writes new raster.
    It also creates a geopandas dataframe of footprint polygons

    Parameters
    ----------
    filename (str)
    in_path (str)
    out_path (str)
    prefix (str) ex. burned_area

    Returns
    -------
    geopandas dataframe with footprint polygon
    """

    #open tif with gdal and convert to array
    # print(f'input raster: {in_path+filename}')

    ds = gdal.Open(in_path+filename)
    gt = ds.GetGeoTransform()
    proj = ds.GetProjection()
    band = ds.GetRasterBand(1)
    array = band.ReadAsArray()
    
    # create burn/not burn binary mask

    binmask = np.where((array > 0),1,0)  # keep all the values that are greater than 0

    # export
    driver = gdal.GetDriverByName("GTiff")
    driver.Register()
    
    bin_filename = f"{prefix}-{filename[4:]}"
    # print(f'output raster: {bin_filename}')
    outds = driver.Create(out_path+bin_filename, xsize = binmask.shape[1],
                      ysize = binmask.shape[0], bands = 1, 
                      eType = gdal.GDT_Int16)
    outds.SetGeoTransform(gt)
    outds.SetProjection(proj)
    outband = outds.GetRasterBand(1)
    outband.WriteArray(binmask)
    outband.SetNoDataValue(np.nan)
    outband.FlushCache()

    # close your datasets and bands!
    outband = None
    outds = None
    
    #open bin_mask and polygonize it
    bin_ = xr.open_rasterio(out_path+bin_filename).squeeze('band', drop=True)

    polygons = polygonize(bin_)
    perimeter = polygons[polygons['value']==1.0]  # select outside polygon
    
    # returns polygon in geodataframe
    perimeter['filename'] = filename
    return perimeter

def make_damage_response_rasters(filename,in_path,out_path, prefix):

    """
    This script takes fire footprint raster files for flame length and maps the values to 
    building damage response values (0, 25, 40, 55, 70, 85, 100) and writes a new raster

    inputs: 
    filename (str)
    in_path (str)
    out_path (str)
    prefix (str)

    outputs:
    damage response rasters
    """
   
    #open tif with gdal and convert to array
    # print(f'input raster: {in_path+filename}')
    ds = gdal.Open(in_path+filename)
    gt = ds.GetGeoTransform()
    proj = ds.GetProjection()
    band = ds.GetRasterBand(1)
    array = band.ReadAsArray()
    
    # map flame length to building damage response values
    # approach as per the Wildfire Risk to Communities paper, https://www.fs.usda.gov/rds/archive/Catalog/RDS-2020-0016
    
    #0' flame --> 0 building response function value
    #0' < flame < 2' --> 25 
    #2-4' flame --> 40
    #4-6' flame --> 55
    #6-8' flame --> 70
    #8-12' flame --> 85
    #>12' flame --> 100
    
    array = np.where((array >= 12),100,array)
    array = np.where(((array >= 8) & (array < 12)),85,array)
    array = np.where(((array >= 6) & (array < 8)),70,array)
    array = np.where(((array >= 4) & (array < 6)),55,array)
    array = np.where(((array >= 2) & (array < 4)),40,array)
    array = np.where(((array > 0) & (array < 2)),25,array)
    array = np.where((array == 0),0,array)
    
    #binmask = np.where((array > 0),1,0)  # keep all the values that are greater than 0

    # export
    driver = gdal.GetDriverByName("GTiff")
    driver.Register()
    
    new_filename = f"{prefix}-{filename[-22:]}"
    # bin_filename = f"{prefix}-{filename[9:]}"
    # print(f'output raster: {new_filename}')
    outds = driver.Create(out_path+new_filename, xsize = array.shape[1],
                      ysize = array.shape[0], bands = 1, 
                      eType = gdal.GDT_Int16)
    outds.SetGeoTransform(gt)
    outds.SetProjection(proj)
    outband = outds.GetRasterBand(1)
    outband.WriteArray(array)
    outband.SetNoDataValue(np.nan)
    outband.FlushCache()

    # close your datasets and bands!
    outband = None
    outds = None

    pass

in_path1 = args[7]
out_path1 = args[8]
prefix1='burned_area'
rpi1 = args[5]
lpi1 = args[6]
rpo1 = args[9]
lpo1 = args[8]
gdf = gpd.GeoDataFrame(columns=['feature'],geometry='feature',crs='EPSG:32610')

in_path2 = args[12]
out_path2 = args[13]
prefix2='damage_response'
rpi2 = args[10]
lpi2 = args[11]
rpo2 = args[14]
lpo2 = args[13]

count = 0

print('Calculating burned area, damage response and intensity rasters')

for fn in filenames:
    count += 1
    f = fn+'.tif'
    # print(count, "f:", f)

    # burned_area rasters
    fs.get(rpath=rpi1+f,lpath=lpi1+f)
    new_gdf = make_polygons('toa-'+f, in_path1, out_path1, prefix1)
    os.remove(lpi1+f)
    fs.put(lpath=lpo1+f"{prefix1}-{f}", rpath=rpo1)
    # print(rpo1, type(rpo1), lpo1+f"{prefix1}-{f}", type(lpo1+f"{prefix1}-{f}"))
    os.remove(out_path1+f"{prefix1}-{f}")
    gdf = gdf.append(new_gdf)

    # damage_response rasters
    fs.get(rpath=rpi2+f,lpath=lpi2+f)
    make_damage_response_rasters('flamelen-'+f,in_path2,out_path2,prefix2)
    # fs.put(lpath=lpi2+f, rpath='landscape-optimization/yosemite/yosemite-footprints/yosemite-footprints/intensity/')
    fs.put(lpath=lpi2+f, rpath=args[15])
    os.remove(lpi2+f)
    
    fs.put(lpath=lpo2+f"{prefix2}-{f}", rpath=rpo2)
    os.remove(out_path2+f"{prefix2}-{f}")

print('Saving footprints polygon json')
gdf = gdf.drop(['feature'], axis=1)
gdf = gdf.set_geometry("geometry")
# print(gdf)
gdf.to_file(args[16], driver="GeoJSON")

def groupby_multipoly(df, by, aggfunc="first"):
    
    """
    This function make multipolygons from polygons for fire footprints

    inputs:
    geopandas dataframe with polygons

    outputs:
    geopandas dataframe with multipolygons
    """
    
    data = df.drop(labels=df.geometry.name, axis=1)
    aggregated_data = data.groupby(by=by).agg(aggfunc)

    # Process spatial component
    def merge_geometries(block):
        return MultiPolygon(block.values)

    g = df.groupby(by=by, group_keys=False)[df.geometry.name].agg(
        merge_geometries
    )

    # Aggregate
    aggregated_geometry = gpd.GeoDataFrame(g, geometry=df.geometry.name, crs=df.crs)
    # Recombine
    aggregated = aggregated_geometry.join(aggregated_data)
    return aggregated

print('Finding intersection with MS building tiles json')

# footprint_polygons
footprint_polygons = gpd.read_file(args[16])
footprint_polygons = footprint_polygons.set_crs('EPSG:32610',allow_override=True)

# make multipolygons from polygons for fire footprints
grouped = groupby_multipoly(footprint_polygons, by='filename').reset_index()

# load bldgs
bldgs = gpd.read_file(args[17]).to_crs('EPSG:32610') 

# intersect fire footprint MPs with bldgs polygons
intersection = gpd.overlay(grouped, bldgs, how='intersection')

# for given fire footprint, merge (essentially a groupby) bldg polygons into multipolygon
gdf2 = intersection.dissolve(by='filename').reset_index()
# print(gdf2.shape)
gdf2.to_file(args[18], driver='GeoJSON')

zero_rasters = []

def make_raster(filename,shp_filename,in_path,out_path, prefix,out_path_tmp):

    """
    This function takes in damage response rasters and creates binary building 
    damage (tmp) rasters and building damage rasters with varying intensity.

    inputs:
    
    filename (str)
    shp_filename (str)
    in_path (str)
    out_path (str)
    prefix (str)
    out_path_tmp (str)

    outputs:

    raster files
    """
    
    fn_ras = in_path+'damage_response-'+filename
    fs.get(rpath=args[21]+filename,lpath=fn_ras)
    # print('input raster: ',fn_ras) 
    ras_ds = gdal.Open(fn_ras)
    intensity_array = ras_ds.GetRasterBand(1).ReadAsArray()
    driver = ogr.GetDriverByName('ESRI Shapefile')
    vec_ds = driver.Open(shp_filename)
    lyr = vec_ds.GetLayer() 
    geot = ras_ds.GetGeoTransform()
    geo_proj = ras_ds.GetProjection() 
    
    # Setup the New Raster
    drv_tiff = gdal.GetDriverByName("GTiff") 
    out_net=f'{out_path_tmp+prefix}-binary-{filename}'
    # print(out_net)
    chn_ras_ds = drv_tiff.Create(out_net, ras_ds.RasterXSize, ras_ds.RasterYSize, 1, gdal.GDT_Float32)
    chn_ras_ds.SetGeoTransform(geot)
    chn_ras_ds.SetProjection(geo_proj) 
    gdal.RasterizeLayer(chn_ras_ds, [1], lyr, burn_values=[1])
    chn_ras_ds.GetRasterBand(1).SetNoDataValue(np.nan) 
    chn_ras_ds = None
    
    # open binary bldg damage raster back up
    ds = gdal.Open(out_net)
    gt = ds.GetGeoTransform()
    proj = ds.GetProjection()
    band = ds.GetRasterBand(1)
    bin_array = band.ReadAsArray()
    
    # get array and use as mask to get values from damage_response raster array
    damage_array = intensity_array*bin_array.astype(int)
    
    #print(damage_array.shape)
    
    # write new raster using chn_ras_ds.GetRasterBand(1).WriteArray(binmask)
    # export
    driver_ = gdal.GetDriverByName("GTiff")
    driver_.Register()
    out=f'{out_path+prefix}-{filename}'
    # print(f'output raster: {out}')
    outds = driver_.Create(out, xsize = damage_array.shape[1],
                      ysize = damage_array.shape[0], bands = 1, 
                      eType = gdal.GDT_Int16)
    outds.SetGeoTransform(gt)
    outds.SetProjection(proj)
    outband = outds.GetRasterBand(1)
    outband.WriteArray(damage_array)
    outband.SetNoDataValue(np.nan)
    outband.FlushCache()

    # close datasets and bands
    outband = None
    outds = None

    fs.put(lpath=args[19]+f"{prefix}-{filename}", rpath=args[22])

    # Check for zero rasters
    ds = gdal.Open(out)
    gt = ds.GetGeoTransform()
    proj = ds.GetProjection()
    band = ds.GetRasterBand(1)
    array = band.ReadAsArray()
    if array.sum()==0:
        zero_rasters.append(f"{prefix}-{filename}")
    # close datasets and bands
    band = None
    ds = None

    os.remove(fn_ras)
    os.remove(out)

    pass

print('Calculating building damage rasters')

in_path = args[12]
out_path = args[19]
prefix='building_damage'
out_path_tmp = args[20]
count = 0
for i in range(len(gdf2)):
    count += 1
    #create shp files
    shp_filename=f"{out_path_tmp+prefix}-{gdf2.iloc[i].filename[4:22]}.shp"
    gdf2.iloc[[i]].to_file(driver = 'ESRI Shapefile', filename=shp_filename)
    root_filename=f"{gdf2.iloc[i].filename[4:22]}.tif"
    # print("filenames", shp_filename, root_filename, count)
    # print(count, "f:", root_filename)
    make_raster(root_filename,shp_filename,in_path,out_path, prefix,out_path_tmp) 
    shutil.rmtree(out_path_tmp, ignore_errors=True)
    os.mkdir(out_path_tmp)

print('# of zero rasters: ', len(zero_rasters))
print('# of files checked: ', len(gdf2))

def groupby_multipoly2(df, by, aggfunc="first"):
    
    """
    This function make multipolygons from polygons for fire footprints

    inputs:
    geopandas dataframe with polygons

    outputs:
    geopandas dataframe with multipolygons
    """    
    
    data = df.drop(labels=df.geometry.name, axis=1)
    aggregated_data = data.groupby(by=by).agg(aggfunc)

    # Process spatial component
    def merge_geometries(block):
        return MultiPolygon(block.values)

    g = df.groupby(by=by, group_keys=False)[df.geometry.name].agg(
        merge_geometries
    )

    # Aggregate
    aggregated_geometry = gpd.GeoDataFrame(g, geometry=df.geometry.name, crs=df.crs)
    # Recombine
    aggregated = aggregated_geometry.join(aggregated_data)
    return aggregated

def make_raster2(filename,shp_filename,in_path,out_path, prefix):
    """
    This function writes raster files

    parameters:
    
        filename (str)
        shp_filename (str)
        in_path (str)
        out_path (str)
        prefix (str)

    outputs:

        raster files
    """
    
    fn_ras = in_path+'toa-'+filename
    # print(filename, fn_ras)
    fs.get(rpath=args[5]+filename,lpath=fn_ras)
    ras_ds = gdal.Open(fn_ras)
    driver = ogr.GetDriverByName('ESRI Shapefile')
    vec_ds = driver.Open(shp_filename) 
    lyr = vec_ds.GetLayer() 
    geot = ras_ds.GetGeoTransform()
    geo_proj = ras_ds.GetProjection() 
    
    # Setup the New Raster
    drv_tiff = gdal.GetDriverByName("GTiff") 
    # out_net=out_path+prefix+filename
    out_net=f'{out_path+prefix}-{filename}'
    # print('writing output raster: ',out_net)
    chn_ras_ds = drv_tiff.Create(out_net, ras_ds.RasterXSize, ras_ds.RasterYSize, 1, gdal.GDT_Float32)
    chn_ras_ds.SetGeoTransform(geot)
    chn_ras_ds.SetProjection(geo_proj) 
    gdal.RasterizeLayer(chn_ras_ds, [1], lyr, burn_values=[1])
    chn_ras_ds.GetRasterBand(1).SetNoDataValue(np.nan) 
    chn_ras_ds = None

    fs.put(lpath=args[24]+f"{prefix}-{filename}", rpath=args[26])
    os.remove(fn_ras)
    os.remove(out_net)
    pass

print('Finding intersection with critical habitat polygons')
# footprint_polygons
footprint_polygons = gpd.read_file(args[16])
footprint_polygons = footprint_polygons.set_crs('EPSG:32610',allow_override=True)

# make multipolygons from polygons for fire footprints
grouped = groupby_multipoly2(footprint_polygons, by='filename').reset_index()

# load critical habitat
habitat = gpd.read_file(args[23]).to_crs('EPSG:32610') 

# intersect fire footprint MPs with habitat polygons
intersection = gpd.overlay(grouped, habitat, how='intersection')

# for given fire footprint, merge (essentially a groupby) bldg polygons into multipolygon
gdf3 = intersection.dissolve(by='filename').reset_index()
# print(gdf3.shape)
gdf2.to_file(args[24], driver='GeoJSON')

print('Calculating habitat damage rasters')
in_path = args[7]
out_path = args[25]
prefix='habitat_damage'
out_path_tmp = args[20]
count = 0
for i in range(len(gdf3)):
    count += 1
    #create shp files
    shp_filename=f"{out_path_tmp+prefix}-{gdf3.iloc[i].filename[4:22]}.shp"
    gdf3.iloc[[i]].to_file(driver = 'ESRI Shapefile', filename=shp_filename)
    root_filename=f"{gdf3.iloc[i].filename[4:22]}.tif"
    # print("filenames", shp_filename, root_filename)
    # print(count, "f:", root_filename)
    make_raster2(root_filename,shp_filename,in_path,out_path, prefix)
    shutil.rmtree(out_path_tmp, ignore_errors=True)
    os.mkdir(out_path_tmp)

