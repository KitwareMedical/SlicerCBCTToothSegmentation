import logging
import os
from typing import Annotated, Optional

import vtk, pathlib, sitkUtils

import slicer
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import (
    parameterNodeWrapper,
    WithinRange,
)

from slicer import vtkMRMLScalarVolumeNode,vtkMRMLSegmentationNode, vtkMRMLMarkupsROINode
from packaging import version

#
# CBCTToothSegmentation
#

class CBCTToothSegmentation(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("CBCTToothSegmentation")  # TODO: make this more human readable by adding spaces
        # TODO: set categories (folders where the module shows up in the module selector)
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "Segmentation")]
        self.parent.dependencies = []  # TODO: add here list of module names that this module requires
        self.parent.contributors = ["John Doe (AnyWare Corp.)"]  # TODO: replace with "Firstname Lastname (Organization)"
        # TODO: update with short description of the module and a link to online module documentation
        # _() function marks text as translatable to other languages
        self.parent.helpText = _("""
This is an example of scripted loadable module bundled in an extension.
See more information in <a href="https://github.com/organization/projectname#CBCTToothSegmentation">module documentation</a>.
""")
        # TODO: replace with organization, grant and thanks
        self.parent.acknowledgementText = _("""
This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc., Andras Lasso, PerkLab,
and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
""")

        # Additional initialization step after application startup is complete
        #slicer.app.connect("startupCompleted()", registerSampleData)


#
# Register sample data sets in Sample Data module
#

def registerSampleData():
    """
    Add data sets to Sample Data module.
    """
    # It is always recommended to provide sample data for users to make it easy to try the module,
    # but if no sample data is available then this method (and associated startupCompeted signal connection) can be removed.

    import SampleData
    iconsPath = os.path.join(os.path.dirname(__file__), 'Resources/Icons')

    # To ensure that the source code repository remains small (can be downloaded and installed quickly)
    # it is recommended to store data sets that are larger than a few MB in a Github release.

    # CBCTToothSegmentation1
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category='CBCTToothSegmentation',
        sampleName='CBCTToothSegmentation1',
        # Thumbnail should have size of approximately 260x280 pixels and stored in Resources/Icons folder.
        # It can be created by Screen Capture module, "Capture all views" option enabled, "Number of images" set to "Single".
        thumbnailFileName=os.path.join(iconsPath, 'CBCTToothSegmentation1.png'),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        fileNames='CBCTToothSegmentation1.nrrd',
        # Checksum to ensure file integrity. Can be computed by this command:
        #  import hashlib; print(hashlib.sha256(open(filename, "rb").read()).hexdigest())
        checksums='SHA256:998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95',
        # This node name will be used when the data set is loaded
        nodeNames='CBCTToothSegmentation1'
    )

    # CBCTToothSegmentation2
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category='CBCTToothSegmentation',
        sampleName='CBCTToothSegmentation2',
        thumbnailFileName=os.path.join(iconsPath, 'CBCTToothSegmentation2.png'),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        fileNames='CBCTToothSegmentation2.nrrd',
        checksums='SHA256:1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97',
        # This node name will be used when the data set is loaded
        nodeNames='CBCTToothSegmentation2'
    )


#
# CBCTToothSegmentationParameterNode
#

@parameterNodeWrapper
class CBCTToothSegmentationParameterNode:
    """
    The parameters needed by module.

    inputVolume - The volume to segment
    """
    inputVolume: vtkMRMLScalarVolumeNode
    outputSegmentation: vtkMRMLSegmentationNode
    inputROI: vtkMRMLMarkupsROINode
    saveSegmentation: vtkMRMLSegmentationNode

#
# CBCTToothSegmentationWidget
#

class CBCTToothSegmentationWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None) -> None:
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._parameterNodeGuiTag = None

    def setup(self) -> None:
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer)
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath('UI/CBCTToothSegmentation.ui'))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = CBCTToothSegmentationLogic()

        # Connections
        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        self.ui.inputROIBox.currentNodeChanged.connect(self.initializeROI)

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()

        # Only show some segment editor effects
        self.ui.SegmentEditorWidget.setEffectNameOrder(["Paint", "Draw", "Erase"])
        self.ui.SegmentEditorWidget.unorderedEffectsVisible = False

        #Set up segment editor
        segmentEditorNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentEditorNode")
        self.ui.SegmentEditorWidget.setMRMLSegmentEditorNode(segmentEditorNode)

        # Buttons
        self.ui.applyButton.connect('clicked(bool)', self.onApplyButton)


    def initializeROI(self) -> None:

        """
        Initialize an ROI to fit the volume
        """
        if self.ui.inputROIBox.currentNode() is None:

            self.ui.adjustROI.enabled = False

        else:
            # Define an initial ROI set to the size of the input image
            crop_module = slicer.vtkMRMLCropVolumeParametersNode()
            slicer.mrmlScene.AddNode(crop_module)
            crop_module.SetInputVolumeNodeID(self.ui.inputSelectorBox.currentNode().GetID())
            # Set output volume as the same volume to overwrite original volume (only needed if you actually want to crop the volume)
            crop_module.SetOutputVolumeNodeID(self.ui.inputSelectorBox.currentNode().GetID())
            # Set the input ROI
            crop_module.SetROINodeID(self.ui.inputROIBox.currentNode().GetID())

            # Use the Fit ROI to Volume function of the crop volume module
            slicer.modules.cropvolume.logic().SnapROIToVoxelGrid(crop_module)  # optional (rotates the ROI to match the volume axis directions)
            slicer.modules.cropvolume.logic().FitROIToInputVolume(crop_module)

            slicer.mrmlScene.RemoveNode(crop_module)
            self.ui.inputROIBox.currentNode().SetName(
                slicer.mrmlScene.GenerateUniqueName("Tooth Segmentation ROI")
            )

            self.ui.adjustROI.enabled = True
            self.ui.adjustROI.setMRMLMarkupsNode(self.ui.inputROIBox.currentNode())


    def cleanup(self) -> None:
        """
        Called when the application closes and the module widget is destroyed.
        """
        self.removeObservers()

    def enter(self) -> None:
        """
        Called each time the user opens this module.
        """
        # Make sure parameter node exists and observed
        self.initializeParameterNode()

    def exit(self) -> None:
        """
        Called each time the user opens a different module.
        """
        # Do not react to parameter node changes (GUI will be updated when the user enters into the module)
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self._parameterNodeGuiTag = None
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)

    def onSceneStartClose(self, caller, event) -> None:
        """
        Called just before the scene is closed.
        """
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event) -> None:
        """
        Called just after the scene is closed.
        """
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self) -> None:
        """
        Ensure parameter node exists and observed.
        """
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.

        self.setParameterNode(self.logic.getParameterNode())

        # Select default input nodes if nothing is selected yet to save a few clicks for the user
        if not self._parameterNode.inputVolume:
            firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
            if firstVolumeNode:
                self._parameterNode.inputVolume = firstVolumeNode

        # Check input ROI node
        if self.ui.inputROIBox.currentNode() is not None:
            self.ui.adjustROI.enabled = True
            self.ui.adjustROI.setMRMLMarkupsNode(self.ui.inputROIBox.currentNode())
        

    def setParameterNode(self, inputParameterNode: Optional[CBCTToothSegmentationParameterNode]) -> None:
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """

        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
        self._parameterNode = inputParameterNode
        if self._parameterNode:
            # Note: in the .ui file, a Qt dynamic property called "SlicerParameterName" is set on each
            # ui element that needs connection.
            self._parameterNodeGuiTag = self._parameterNode.connectGui(self.ui)
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._canExportFile)
            self._checkCanApply()
            self._canExportFile()

    def _canExportFile(self, caller = None, event = None) -> None:
        if self._parameterNode and self._parameterNode.saveSegmentation:
            self.ui.FileExportWidget.setSegmentationNode(self._parameterNode.saveSegmentation)

    def _checkCanApply(self, caller=None, event=None) -> None:
        if self._parameterNode and self._parameterNode.inputVolume and self._parameterNode.inputROI:
            self.ui.applyButton.toolTip = _("Compute output tooth segmentation")
            self.ui.applyButton.enabled = True
        else:
            self.ui.applyButton.toolTip = _("Select input volume node and input seed")
            self.ui.applyButton.enabled = False

    def onApplyButton(self) -> None:
        """
        Run processing when user clicks "Apply" button.
        """
        with slicer.util.tryWithErrorDisplay(_("Failed to compute results."), waitCursor=True):
            
            #segmentationNode = self.ui.outputSelectorBox.currentNode()
            # Create output segmentation node, if not created yet 
            outputSegmentationNode = self._parameterNode.outputSegmentation
            if not outputSegmentationNode:
                outputSegmentationNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLSegmentationNode', slicer.mrmlScene.GenerateUniqueName('Tooth'))
                outputSegmentationNode.CreateDefaultDisplayNodes()
                outputSegmentationNode.SetReferenceImageGeometryParameterFromVolumeNode(self.ui.inputSelectorBox.currentNode())
                self.ui.outputSelectorBox.setCurrentNode(outputSegmentationNode)

            # Setup python requirements
            self.logic.setupPythonRequirements()

            #self.ui.inputSelectorBox.currentNode()
            # Compute output segmentation
            self.logic.process(self._parameterNode.inputVolume, outputSegmentationNode, self.ui.inputROIBox.currentNode())

            #Un-collapse the Segmentation Editor
            self.ui.SegmentEditorButton.collapsed = False

            self.logic.initializeFinalSteps(self.ui.inputSelectorBox.currentNode(), outputSegmentationNode)
# CBCTToothSegmentationLogic
#

class CBCTToothSegmentationLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self) -> None:
        """
        Called when the logic class is instantiated. Can be used for initializing member variables.
        """
        ScriptedLoadableModuleLogic.__init__(self)

    def getParameterNode(self):
        return CBCTToothSegmentationParameterNode(super().getParameterNode())

    def setupPythonRequirements(self):

        print('Checking python dependencies')

        # Install itk.
        try:
            import itk
        except:
            logging.debug('Module requires the itk Python package. Installing... (it may take several minutes)')
            slicer.util.pip_install('itk')     

        # Install PyTorch
        try:
            import PyTorchUtils
        except ModuleNotFoundError:
            slicer.util.messageBox("Module requires the PyTorch extension. Please install it from the Extensions Manager.")
        
        torchLogic = PyTorchUtils.PyTorchUtilsLogic()
        torchInstalled = torchLogic.torchInstalled()
        if not torchInstalled:
            logging.debug('Module requires the PyTorch Python package. Installing... (it may take several minutes)')
            torch = torchLogic.installTorch(askConfirmation=True)
            if torch is None:
                slicer.util.messageBox('PyTorch extension needs to be installed manually to use this module.')
        else:
            logging.debug("Found torch already installed")
    
        # import torch
    
        # Install MONAI and restart if the version was updated.
        monaiVersion = "1.3.0"
        try:
            import monai
            if version.parse(monai.__version__) != version.parse(monaiVersion):
                logging.debug(f'Module requires MONAI version {monaiVersion}. Installing... (it may take several minutes)')
                slicer.util.pip_uninstall('monai')
                slicer.util.pip_install('monai[pynrrd,fire]=='+ monaiVersion)
                if slicer.util.confirmOkCancelDisplay(f'MONAI version was updated {monaiVersion}.\n Click OK restart Slicer.'):
                    slicer.util.restart()
        except:
            logging.debug('Module requires the MONAI Python package. Installing... (it may take several minutes)')
            slicer.util.pip_install('monai[pynrrd,fire]=='+ monaiVersion)
    

    def removeModel(self, modelPath) -> None:
        os.remove(modelPath)

    def downloadModel(self) -> None:

        """
        Download model from shared google drive link
        """

        import SampleData
        import os, qt


        url = "https://drive.google.com/uc?export=download&id=1J2DEBGKxbNWJ6KM2dZ8NTI3KaoAF3q4e"
        #V1: url = "https://drive.usercontent.google.com/download?id=1a0uxBQmVkCAaqKNJiFEo3NKAfCv5z0SL&export=download&authuser=0"
        destination_folder = qt.QDir().tempPath()
        modelPath = os.path.join(destination_folder,'pretrainedmodel-DentalCBCTSegmentation.pth')
        name = 'pretrainedmodel-DentalCBCTSegmentation.pth'
        if not os.path.exists(modelPath):
            print("Downloading pretrained model to local temp directory...")
            print("...")
            SampleData.SampleDataLogic().downloadFile(url, destination_folder, name, checksum=None)
            print('Done.')
        print('Pre-trained model saved to: ', modelPath)
        
        return modelPath
        
    def initializeFinalSteps(self,inputVolume: vtkMRMLScalarVolumeNode,
                                outputSegmentationNode: vtkMRMLSegmentationNode) -> None:

        slicer.modules.CBCTToothSegmentationWidget.ui.SegmentEditorWidget.setSegmentationNode(outputSegmentationNode)
        slicer.modules.CBCTToothSegmentationWidget.ui.SegmentEditorWidget.setSourceVolumeNode(inputVolume)


    def process(self,
                inputVolume: vtkMRMLScalarVolumeNode,
                outputSegmentationNode: vtkMRMLSegmentationNode,
                inputROI: vtkMRMLMarkupsROINode) -> None:
        """
        Run the processing algorithm.
        Can be used without GUI widget.
        :param inputVolume: volume to be Segmented
        :param outputVolume: Segmentation result
        :param inputROI - To ADD
        :param showResult: show output volume in slice viewers
        """

        if not inputVolume or not outputSegmentationNode:
            raise ValueError("Input or output volume is invalid")

        import time
        startTime = time.time()
        logging.info('Processing started')

        # ## Get the spacing of the input volume
        inputSpacing = inputVolume.GetSpacing()

        # # Caculate spacing scale
        targetSpacing = 0.0075
        spacingScale = [0,0,0]
        for i in range(3):
            spacingScale[i] = targetSpacing/inputSpacing[i]
        print('Upsampling image using scaling factor: ', spacingScale)

        # Crop the image using the user specified ROI
        cropVolumeLogic = slicer.modules.cropvolume.logic()
        cropVolumeParameterNode = slicer.vtkMRMLCropVolumeParametersNode()
        cropVolumeParameterNode.SetROINodeID(inputROI.GetID())
        cropVolumeParameterNode.SetInputVolumeNodeID(inputVolume.GetID())
        cropVolumeParameterNode.SetVoxelBased(True)

        # Create cropped output volume node
        croppedVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "croppedVolume")

        cropVolumeLogic.CropInterpolated(inputROI, inputVolume, croppedVolume, True, max(spacingScale), 0, 0.0)
        #cropVolumeLogic.Apply(cropVolumeParameterNode)
        # croppedVolume = slicer.mrmlScene.GetNodeByID(cropVolumeParameterNode.GetOutputVolumeNodeID())

        slicer.util.exportNode(croppedVolume,"cropped_region.nrrd")
    

        inputImageArray  = slicer.util.arrayFromVolume(croppedVolume)
        inputCrop_shape = inputImageArray.shape
        print('Network input shape:', inputCrop_shape)
       
        # Automated segmentation
        #Import MONAI and dependencies
        import numpy as np
        import torch
        from monai.inferers import SlidingWindowInferer

        from monai.transforms import (
          Compose,
          EnsureChannelFirst,
          SpatialPad,
          NormalizeIntensity
        )
        from monai.networks.nets import UNet
        from monai.networks.layers.factories import Act
        from monai.networks.layers import Norm

        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = "cpu"
        
        print("Using ", device, " for compute")
    
        #Define U-Net model
        model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            channels=(16,32,64,128),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            act=Act.RELU,
            norm=Norm.BATCH,
            dropout=0.2).to(device)
        
        #Load model weights
        inputModelPath = self.downloadModel()
        loaded_model = torch.load(inputModelPath, map_location=device)
        model.load_state_dict(loaded_model, strict = True) #Strict is false since U-Net is missing some keys - batch norm related?
        model.eval()

        inputImageArray = torch.tensor(inputImageArray, dtype=torch.float)

        # define pre-transforms
        pre_transforms = Compose([
            EnsureChannelFirst(channel_dim='no_channel'),
            NormalizeIntensity(),
            SpatialPad(spatial_size = [144,144,144], mode= "reflect"),
            EnsureChannelFirst(channel_dim='no_channel') 
        ])

        # run inference
        inputProcessed = pre_transforms(inputImageArray).to(device)
        inferer = SlidingWindowInferer(roi_size=[96,96,96])

        # Delete downloaded model from temp directory
        self.removeModel(inputModelPath)

        # process prediction output
        output = inferer(inputProcessed, model)
        output = torch.softmax(output, axis=1).data.cpu().numpy()
        output = np.argmax(output, 1).squeeze().astype(np.uint8)

        # Crop the predicion back to original size
        lower = [0]*3
        upper = [0]*3
        for i in range(len(inputCrop_shape)):
            dim = inputCrop_shape[i]
            padding = 144 - dim
            if padding > 0:
                lower[i] = int(np.floor(padding/2))
                upper[i] = -int(np.ceil(padding/2))
            else:
                lower[i] = 0
                upper[i] = dim
      
        output_reshaped = output[lower[0]:upper[0],lower[1]:upper[1],lower[2]:upper[2]]

        # # Keep largest connected component
        # largest_comp_transform = KeepLargestConnectedComponent()
        # val_comp = largest_comp_transform(val_outputs)                                                                              

        print("Inference done")

        # Need to take cropped segmentation back into the space of the original image croppedVolume
        outputSegmentationNode.SetReferenceImageGeometryParameterFromVolumeNode(croppedVolume)
        outputSegmentationNode.GetSegmentation().AddEmptySegment("ToothSegmentation")
        segmentId =  outputSegmentationNode.GetSegmentation().GetSegmentIDs()[-1]

        slicer.util.updateSegmentBinaryLabelmapFromArray(output_reshaped, outputSegmentationNode, segmentId)

        # set reference geometry to match input volume
        outputSegmentationNode.SetReferenceImageGeometryParameterFromVolumeNode(inputVolume)

        slicer.mrmlScene.RemoveNode(croppedVolume)

        slicer.util.setSliceViewerLayers(background=inputVolume,foreground = outputSegmentationNode, foregroundOpacity=0.5)

        stopTime = time.time()
        logging.info(f'Processing completed in {stopTime-startTime:.2f} seconds')

#
# CBCTToothSegmentationTest
#

class CBCTToothSegmentationTest(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        """ Do whatever is needed to reset the state - typically a scene clear will be enough.
        """
        slicer.mrmlScene.Clear()

    def runTest(self):
        """Run as few or as many tests as needed here.
        """
        self.setUp()
        self.test_CBCTToothSegmentation1()

    def test_CBCTToothSegmentation1(self):
        """ Ideally you should have several levels of tests.  At the lowest level
        tests should exercise the functionality of the logic with different inputs
        (both valid and invalid).  At higher levels your tests should emulate the
        way the user would interact with your code and confirm that it still works
        the way you intended.
        One of the most important features of the tests is that it should alert other
        developers when their changes will have an impact on the behavior of your
        module.  For example, if a developer removes a feature that you depend on,
        your test should break so they know that the feature is needed.
        """

        self.delayDisplay("Starting the test")

        # Get/create input data

        import SampleData
        registerSampleData()
        inputVolume = SampleData.downloadSample('CBCTToothSegmentation1')
        self.delayDisplay('Loaded test data set')

        inputScalarRange = inputVolume.GetImageData().GetScalarRange()
        self.assertEqual(inputScalarRange[0], 0)
        self.assertEqual(inputScalarRange[1], 695)

        outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
        threshold = 100

        # Test the module logic

        logic = CBCTToothSegmentationLogic()

        # Test algorithm with non-inverted threshold
        logic.process(inputVolume, outputVolume, threshold, True)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], threshold)

        # Test algorithm with inverted threshold
        logic.process(inputVolume, outputVolume, threshold, False)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], inputScalarRange[1])

        self.delayDisplay('Test passed')
