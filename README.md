# MSI Community Detection (GRINE)
MSI community detection is part of the [GRINE project](https://github.com/Kawue/grine-v2). The project can be used to detect communities in image networks build from mass spectrometry imaging data. It will produce a .JSON file, which is needed to start GRINE. If you use GRINE or MSI Community Detection for your own work, please reference it as [1].

## Docker Version
Start Docker, navigate into the msi-community-detection directory and call:
`docker build -t grine/msicommunitydetection .`

### Usage
To start the script call:
`docker run --rm grine/msicommunitydetection`
For information about the required command line parameter use `-h`.
The resulting data files can be can be used within GRINE.

The script needs the MSI data to be in HDF5 format.
In case of raw data we refer to our preprocessing pipeline [A Mad Pie](https://github.com/Kawue/amadpie/).
In case of processed data we refer to our parser [imzML-to-HDF5-Parser](https://github.com/Kawue/imzML-to-HDF5).

## Local Version (without Docker)
Install Anaconda, use the environment file to create the grine conda environment manually and call `main.py`.


## Docker on Windows 7, 8 and 10 Home
1. Visit https://docs.docker.com/toolbox/toolbox_install_windows/. Follow the given instructions and install all required software to install the Docker Toolbox on Windows.
2. Control if virtualization is enabled on your system. Task Manager -> Performance tab -> CPU -> Virtualization. If it is enabled continue with Step X.
3. If virtualization is disabled, it needs to be enabled in your BIOS. Navigate into your systems BIOS and look for Virtualization Technology (VT). Enable VT, save settings and reboot. This option is most likely part of the Advanced or Security tab. This step can deviate based on your Windows and Mainboard Manufacturer.
4. Open your CMD as administrator and call `docker-machine create default --virtualbox-no-vtx-check`. A restart may be required.
5. In your OracleVM VirtualBox selected the appropriate machine (probably the one labeled "default") -> Settings -> Network -> Adapter 1 -> Advanced -> Port Forwarding. Click on "+" to add a new rule and set Host Port to 8080 and Guest Port to 8080. Be sure to leave Host IP and Guest IP empty. Also, add another rule for the Port 5000 in the same way. A restart of your VirtualBox may be needed.
6. Now you should be ready to use the Docker QuickStart Shell to call the Docker commands provided to start this tool.




[1]: https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-019-2890-6
