FROM ubuntu

RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y software-properties-common

RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update
RUN apt-get install -y python3.9 python3-pip python3.9-venv git gdal-bin libgdal-dev 
RUN apt-get install -y python3.9-dev

# Enable venv
RUN python3.9 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip3 install pymoo rasterio shapely numpy==1.24 pandas matplotlib geopandas
RUN pip3 install pyyaml rasterio rioxarray
RUN pip3 install dask dask[dataframe] dask[distributed]
RUN pip3 install gdal==3.8.4

RUN mkdir /landopt
COPY landscape-optimization /landopt/landscape-optimization 
COPY data /landopt/data
COPY preprocess.sh /landopt/preprocess.sh
COPY run.sh /landopt/run.sh

# Enable venv
ENV PATH="/opt/venv/bin:$PATH"
