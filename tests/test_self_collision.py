import json

import torch

import pytorch_kinematics as pk
from pytorch_collision_checker import cfg
from pytorch_collision_checker.collision_checker import CollisionChecker, get_default_ignores

ANT_PATH = cfg.TEST_DIR / "ant.xml"
ANT_SPHERES_PATH = cfg.TEST_DIR / "ant_spheres.json"


def test_self_collision_mjcf():
    torch.set_printoptions(linewidth=250, precision=2, sci_mode=0)
    chain = pk.build_chain_from_mjcf(ANT_PATH.open().read())
    spheres = load_spheres(ANT_SPHERES_PATH)
    dtype = torch.float64
    device = 'cuda'
    chain = chain.to(dtype=dtype, device=device)

    # env = torch.tensor

    ignore = get_default_ignores(chain, spheres)
    cc = CollisionChecker(chain, spheres, env=None, ignore_collision_pairs=ignore)
    from time import perf_counter
    for b in [1, 10, 100, 1000, 10_000, 100_000]:
        q = torch.randn([b, 8], dtype=dtype, device=device)
        t0 = perf_counter()
        cc.check_collision(q)
        dt = perf_counter() - t0
        print(f'{b:9d} dt: {1000 * dt:.1f}ms')


def load_spheres(path):
    with path.open('r') as f:
        spheres = json.load(f)['spheres']
    return spheres


if __name__ == '__main__':
    test_self_collision_mjcf()
