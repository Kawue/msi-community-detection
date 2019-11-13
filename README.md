# MSI Community Detection (GRINE)
MSI community detection is part of the [GRINE project](https://github.com/Kawue/grine-v2). The project can be used to detect communities in image networks build from mass spectrometry imaging data. It will produce a .JSON file, which is needed to start GRINE. If you use GRINE or MSI Community Detection for your own work, please reference it as [1].

## Usage
Install the provided anaconda environment with:
```
conda env create -f environment.yml
```
Therafter activate the environment via:
```
conda activate grine
```
And call:
```
python main.py -h
```
to get all needed parameter information.

If your data is in `.imzML` and `.ibd` format instead of HDF5 we refer to our preprocessing pipeline [A Mad Pie](https://github.com/Kawue/amadpie/) for a proper conversion. For a self written processing we refer to [pyImzML-Parser](https://github.com/alexandrovteam/pyimzML/blob/master/pyimzml/ImzMLParser.py) and [Pandas](https://pandas.pydata.org/) .


[1]: https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-019-2890-6
