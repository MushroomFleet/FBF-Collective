NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

try:
    from .AudioToFrames import AudioToFrames
    NODE_CLASS_MAPPINGS["AudioToFrames"] = AudioToFrames
    NODE_DISPLAY_NAME_MAPPINGS["AudioToFrames"] = "Audio To Frames"
except ImportError:
    print("Unable to import AudioToFrames. This node will not be available.")

try:
    from .AudioScratchV2 import AudioScratchV2
    NODE_CLASS_MAPPINGS["Sonification:_Audio_Scratch_V2"] = AudioScratchV2
    NODE_DISPLAY_NAME_MAPPINGS["Sonif_2"] = "AudioScratchV2"
except ImportError:
    print("Unable to import AudioScratchV2. This node will not be available.")

try:
    from .DjzDatamoshV8Enhanced import DjzDatamoshV8Enhanced
    NODE_CLASS_MAPPINGS["DjzDatamoshV8Enhanced"] = DjzDatamoshV8Enhanced
    NODE_DISPLAY_NAME_MAPPINGS["DjzDatamoshV8Enhanced"] = "Djz Pixel Sort V8 Enhanced"
except ImportError:
    print("Unable to import DjzDatamoshV8Enhanced. This node will not be available.")

try:
    from .Dynamic_Spectrogram import DynamicSpectrogram
    NODE_CLASS_MAPPINGS["DynamicSpectrogram"] = DynamicSpectrogram
    NODE_DISPLAY_NAME_MAPPINGS["DynamicSpectrogram"] = "Dynamic Spectrogram"
except ImportError:
    print("Unable to import DynamicSpectrogram. This node will not be available.")

try:
    from .FrameBufferGlitch import FrameBufferGlitch
    NODE_CLASS_MAPPINGS["FrameBufferGlitch"] = FrameBufferGlitch
    NODE_DISPLAY_NAME_MAPPINGS["FrameBufferGlitch"] = "Frame Buffer Glitch"
except ImportError:
    print("Unable to import FrameBufferGlitch. This node will not be available.")

try:
    from .FrameShuffle import FrameShuffle
    NODE_CLASS_MAPPINGS["FrameShuffle"] = FrameShuffle
    NODE_DISPLAY_NAME_MAPPINGS["FrameShuffle"] = "Frame Shuffle"
except ImportError:
    print("Unable to import FrameShuffle. This node will not be available.")

try:
    from .Signal2Frames import SignalToFrames
    NODE_CLASS_MAPPINGS["SignalToFrames"] = SignalToFrames
    NODE_DISPLAY_NAME_MAPPINGS["SignalToFrames"] = "Signal to Frames"
except ImportError:
    print("Unable to import SignalToFrames. This node will not be available.")

try:
    from .ToadVideoFrameManipulator import ToadVideoFrameManipulator
    NODE_CLASS_MAPPINGS["ToadVideoFrameManipulator"] = ToadVideoFrameManipulator
    NODE_DISPLAY_NAME_MAPPINGS["ToadVideoFrameManipulator"] = "Toad Video Frame Manipulator"
except ImportError:
    print("Unable to import ToadVideoFrameManipulator. This node will not be available.")


try:
    from .Frame_Duplicator import Frame_Duplicator
    NODE_CLASS_MAPPINGS["Frame_Duplicator"] = Frame_Duplicator
    NODE_DISPLAY_NAME_MAPPINGS["Frame_Duplicator"] = "Frame Duplicator"
except ImportError:
    print("Unable to import Frame_Duplicator. This node will not be available.")


try:
    from .LTX_ConDelta import LTX_ConDelta
    NODE_CLASS_MAPPINGS["LTX_ConDelta"] = LTX_ConDelta
    NODE_DISPLAY_NAME_MAPPINGS["LTX_ConDelta"] = "LTX ConDelta"
except ImportError:
    print("Unable to import LTX_ConDelta. This node will not be available.")


try:
    from .MixerSonif import MixerSonif
    NODE_CLASS_MAPPINGS["MixerSonif"] = MixerSonif
    NODE_DISPLAY_NAME_MAPPINGS["ToadVideoFMixerSoniframeManipulator"] = "Mixer Sonif"
except ImportError:
    print("Unable to import MixerSonif. This node will not be available.")


try:
    from .Psych import Psych
    NODE_CLASS_MAPPINGS["Psych"] = Psych
    NODE_DISPLAY_NAME_MAPPINGS["Psych"] = "Audio-Driven RGB Modifier"
except ImportError:
    print("Unable to import Psych. This node will not be available.")


try:
    from .RGB_zoom import RGB_zoom
    NODE_CLASS_MAPPINGS["RGB_zoom"] = RGB_zoom
    NODE_DISPLAY_NAME_MAPPINGS["RGB_zoom"] = "RGB zoom"
except ImportError:
    print("Unable to import RGB_zoom. This node will not be available.")


try:
    from .Video_Echo import Video_Echo
    NODE_CLASS_MAPPINGS["Video_Echo"] = Video_Echo
    NODE_DISPLAY_NAME_MAPPINGS["Video_Echo"] = "Video Echo"
except ImportError:
    print("Unable to import Video_Echo. This node will not be available.")


try:
    from .VideoRGBSplitter import VideoRGBSplitter
    NODE_CLASS_MAPPINGS["VideoRGBSplitter"] = VideoRGBSplitter
    NODE_DISPLAY_NAME_MAPPINGS["VideoRGBSplitter"] = "Video RGB Splitter"
except ImportError:
    print("Unable to import VideoRGBSplitter. This node will not be available.")


try:
    from .VoiceMod import VoiceMod
    NODE_CLASS_MAPPINGS["VoiceMod"] = VoiceMod
    NODE_DISPLAY_NAME_MAPPINGS["VoiceMod"] = "Voice Mod"
except ImportError:
    print("Unable to import VoiceMod. This node will not be available.")


try:
    from .RGBStream import RGBStream
    NODE_CLASS_MAPPINGS["RGBStream"] = RGBStream
    NODE_DISPLAY_NAME_MAPPINGS["RGBStream"] = "RGBStream"
except ImportError:
    print("Unable to import RGBStream. This node will not be available.")


__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
