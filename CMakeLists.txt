cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(DentalCBCTToolkit)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://github.com/KitwareMedical/SlicerCBCTToothSegmentation")
set(EXTENSION_CATEGORY "Segmentation")
set(EXTENSION_CONTRIBUTORS "Sadhana Ravikumar (Kitware Inc.), Jared Vicory (Kitware Inc.), James Fishbaugh (Kitware Inc.), Beatriz Paniagua (Kitware Inc.)")
set(EXTENSION_DESCRIPTION "Automated segmentation of individual teeth in cone-beam CT dental scans using a deep-learning based approach.")
set(EXTENSION_ICONURL "https://github.com/KitwareMedical/SlicerCBCTToothSegmentation/master/CBCTToothSegmentation/Resources/Icons/CBCTToothSegmentation.png")
set(EXTENSION_SCREENSHOTURLS "https://github.com/KitwareMedical/SlicerCBCTToothSegmentation/master/Screenshot01.png")
set(EXTENSION_DEPENDS "NA") # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(CBCTToothSegmentation)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
