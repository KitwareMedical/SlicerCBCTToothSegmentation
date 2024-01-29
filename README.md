SlicerCBCTToothSegmentation
===========================

Slicer extension for automated segmentation of individual teeth in cone-beam CT dental scans using a deep-learning based approach.



Usage
-----

1. Load the CBCT scan to be segmented into slicer and select it as the 'Input volume'.
2. Create a new 'Input ROI' and adjust the bounding box to surround the tooth of interest that you would like to segment. 
3. Specify an 'Output Segmentation' and click the 'Apply' button. 

- Steps 2) and 3) can be repeated to segment multiple teeth. 
- The `segmentation editor` built into the module can be used to edit the automated output segmentation using the `Paint` and `Erase` tools.
- The `Export to file` section can be used to directly save the output segmentation to file. 

License
-------

This extension is covered by the Apache License, Version 2.0:

https://www.apache.org/licenses/LICENSE-2.0

