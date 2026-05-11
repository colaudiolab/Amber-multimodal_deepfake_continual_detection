# -*- coding: utf-8 -*-
from .Learner import Learner
from .Buffer import Buffer, RandomBuffer, GaussianKernel
from .AnalyticLinear import AnalyticLinear, RecursiveLinear
from .ACIL import ACIL, ACILLearner
from .DSAL import DSAL, DSALLearner
from .GKEAL import GKEAL, GKEALLearner
from .AEFOCL import AEFOCL, AEFOCLLearner
from .AIR import AIRLearner, GeneralizedAIRLearner
from .Finetune import FinetuneLearner
from .PASS import PassLearner
from .SSRE import SSRELearner
from .iCaRL import iCaRLLearner
from .Amber import AmberLearner
from .LWF import LWFLearner
from .Fetril import FetrilLearner
from .MMAL import MMALLearner
from .DA_AIL import DAAILLearner
from .ADanser import ADanserLearner
from .Replay import ReplayLearner
from .DFIL import DFILLearner
from .MyTagFex import TagFexLearner
from .xmfewshot import xmfewshotLearner
from .DyCR import DyCRLearner
from .Frex import FrexLearner

__all__ = [
    "Learner",
    "Buffer",
    "RandomBuffer",
    "GaussianKernel",
    "AnalyticLinear",
    "RecursiveLinear",
    "ACIL",
    "DSAL",
    "GKEAL",
    "AEFOCL",
    "ACILLearner",
    "DSALLearner",
    "GKEALLearner",
    "AEFOCLLearner",
    "AIRLearner",
    "GeneralizedAIRLearner",
    "FinetuneLearner",
    "PassLearner",
    "SSRELearner",
    "iCaRLLearner",
    "AmberLearner",
    "LWFLearner",
    "FetrilLearner"
    "MMALLearner",
    "DAAILLearner",
    "ADanserLearner",
    "ReplayLearner",
    "DFILLearner",
    "TagFexLearner",
    "xmfewshotLearner",
    "DyCRLearner",
    "FrexLearner",
]
