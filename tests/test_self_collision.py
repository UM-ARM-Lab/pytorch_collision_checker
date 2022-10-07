import torch

import pytorch_kinematics as pk
from pytorch_collision_checker import cfg
from pytorch_collision_checker.collision_checker import CollisionChecker

ANT_PATH = cfg.TEST_DIR / "ant.xml"


def test_self_collision_mjcf():
    chain = pk.build_chain_from_mjcf(ANT_PATH.open().read())
    dtype = torch.float64
    device = 'cpu'
    chain = chain.to(dtype=dtype)

    # env = torch.tensor
    cc = CollisionChecker(chain, env=None)
    print(cc.check_collision(torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, ], dtype=dtype, device=device)))


if __name__ == '__main__':
    test_self_collision_mjcf()
