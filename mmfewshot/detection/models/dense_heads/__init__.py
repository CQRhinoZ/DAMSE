# Copyright (c) OpenMMLab. All rights reserved.
from .attention_rpn_head import AttentionRPNHead
from .two_branch_rpn_head import TwoBranchRPNHead

from .my_head import my_RPNHead
__all__ = ['AttentionRPNHead', 'TwoBranchRPNHead',
           'my_RPNHead'
           ]
