"""Panda quadruped tasks, organized by task family like the Go2 tasks."""

from .PANDA_Flip.PANDA_BackFlip import (
    PandaBackFlip,
    PandaBackFlipCfg,
    PandaBackFlipCfgPPO,
)
from .PANDA_Flip.PANDA_Spring_Jump import (
    PandaSpringJump,
    PandaSpringJumpCfg,
    PandaSpringJumpCfgPPO,
)
from .PANDA_Stand.PANDA_Handstand import (
    PandaHandstand,
    PandaHandstandCfg,
    PandaHandstandCfgPPO,
)
from .PANDA_Stand.PANDA_Leggedstand import (
    PandaLeggedstand,
    PandaLeggedstandCfg,
    PandaLeggedstandCfgPPO,
)
from .Panda_MoB.PANDA_JUMP import PandaJump, PandaJumpCfg, PandaJumpCfgPPO

