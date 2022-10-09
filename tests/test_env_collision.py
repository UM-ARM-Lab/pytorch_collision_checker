import torch

import pytorch_kinematics as pk
from pytorch_collision_checker import cfg
from pytorch_collision_checker.collision_checker import CollisionChecker, get_default_ignores
from pytorch_collision_checker.sdf import SDF

ANT_PATH = cfg.TEST_DIR / "ant.xml"


def test_env_collision_mjcf():
    torch.set_printoptions(linewidth=250, precision=2, sci_mode=0)
    chain = pk.build_chain_from_mjcf(ANT_PATH.open().read())
    dtype = torch.float64
    device = 'cpu'
    chain = chain.to(dtype=dtype)

    origin_point = torch.tensor([[0.0, 0, 0]])
    res = torch.tensor([[0]])
    sdf = torch.tensor([])
    env = SDF(origin_point, res, sdf)
    radii = torch.tensor([[0.125, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]], dtype=dtype,
                         device=device)

    ignore = get_default_ignores(chain, radii)
    cc = CollisionChecker(chain, radii, env=env, ignore_collision_pairs=ignore)

    from time import perf_counter
    t0 = perf_counter()
    q = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0]], dtype=dtype, device=device)
    print(cc.check_collision(q))
    print(f'dt: {perf_counter() - t0:.5f}')


if __name__ == '__main__':
    test_env_collision_mjcf()
