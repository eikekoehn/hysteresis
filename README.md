# hysteresis

This package provides some functions to calculate hysteresis properties.

For now it can either calculate hysteresis metrics in a 1D case (based on numpy arrays) or along the 'time' dimension in a 3D case ('year','lat','lon' as dimensions), building on xarray. 

An important note: For now, the functions are implemented as such, that they can only handle cases in which the reference axis (e.g. atmospheric CO2, global mean surface temperature) increase/decrease strictly monotonically from the start of the loop to the global maximum and from the global maximum to the end of the loop. Extra care needs to be taken, with potential changes to the code, if this condition is not met. 

To use it, clone this repository to your directory of preference by executing

```
cd /path/to/dir_pref
git clone https://github.com/eikekoehn/hysteresis.git
```

Then, to import the module in your script use:

```
import sys
sys.path.insert(0,'/path/to/dir_pref/')
import hysteresis # import the package
from hysteresis import hyst_areas as ha # import the submodule hyst_areas
```

--- author: Eike E. Köhn
--- date: Mar 20, 2025