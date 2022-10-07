from typing import Optional

import torch

from pytorch_kinematics import Chain, SerialChain


class CollisionChecker:

    def __init__(self, chain: Chain, env: Optional[torch.tensor] = None):
        if isinstance(chain, SerialChain):
            raise NotImplementedError("only Chain supported")
        self.chain = chain
        self.env = env

    def check_collision(self, joint_positions):
        transforms = self.chain.forward_kinematics(joint_positions)
        for link_name, transform in transforms.items():
            print(transform)

        return False
