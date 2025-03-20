# hysteresis

This package provides some functions to calculate hysteresis properties.

An important note: For now, the functions are implemented as such, that they can only handle cases in which the reference axis (e.g. atmospheric CO2, global mean surface temperature) increase/decrease strictly monotonically from the start of the loop to the global maximum and from the global maximum to the end of the loop. Extra care needs to be taken, with potential changes to the code, if this condition is not met. 

To use it, clone this repository to your directory of preference (/path/to/dir_pref).
Then, to import the module in your script use:

```
import sys
sys.path.insert(0,'/path/to/dir_pref/')
import hysteresis # import the package
from hysteresis import hyst_areas as ha # import the submodule hyst_areas
```


--- author: Eike E. Köhn
--- date: Mar 20, 2025

