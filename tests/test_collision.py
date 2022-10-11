import pickle

import torch

import pytorch_kinematics as pk
from pytorch_collision_checker import cfg
from pytorch_collision_checker.collision_checker import CollisionChecker, get_default_ignores, load_spheres

ANT_PATH = cfg.TEST_DIR / "ant.xml"
ANT_SPHERES_PATH = cfg.TEST_DIR / "ant_spheres.json"
ANT_ENV_PATH = cfg.TEST_DIR / "ant_env_sdf.pkl"


def test_self_collision_mjcf():
    chain, device, dtype, ignore, spheres = ant_setup()
    cc = CollisionChecker(chain, spheres, sdf=None, ignore_collision_pairs=ignore)
    print("self-collision")
    evaluate_perf(cc)


def test_sdf_collision_mjcf():
    chain, device, dtype, ignore, spheres = ant_setup()
    with ANT_ENV_PATH.open("rb") as f:
        sdf = pickle.load(f)
    sdf = sdf.to(dtype=dtype, device=device)
    cc = CollisionChecker(chain, spheres, sdf=sdf, ignore_collision_pairs=ignore)
    print("self-collision and env/sdf collision")
    evaluate_perf(cc)


def evaluate_perf(cc):
    from time import perf_counter
    for b in [1, 10, 100, 1000, 10_000, 20_000, 40_000, 80_000]:
        q = torch.randn([b, 8], dtype=cc.dtype, device=cc.device)
        t0 = perf_counter()
        cc.check_collision(q)
        dt = perf_counter() - t0
        print(f'{b:9d} dt: {1000 * dt:.1f}ms')


def ant_setup():
    torch.set_printoptions(linewidth=250, precision=2, sci_mode=0)
    chain = pk.build_chain_from_mjcf(ANT_PATH.open().read())
    spheres = load_spheres(ANT_SPHERES_PATH)
    dtype = torch.float64
    device = 'cpu'
    chain = chain.to(dtype=dtype, device=device)
    ignore = get_default_ignores(chain, spheres)
    return chain, device, dtype, ignore, spheres


if __name__ == '__main__':
    test_self_collision_mjcf()
    test_sdf_collision_mjcf()
