"""环境健康检查脚本。

在 rl-torch 和 rl-jax 两个环境里都跑一遍。脚本会自动检测当前环境装了哪些库，
只测装上的部分，不会因为缺包就整体失败。
"""

import importlib
import platform
import sys
from textwrap import indent


def check(name: str, fn) -> bool:
    """运行一个检查项，返回是否通过。"""
    print(f"[ ] {name} ...", end=" ", flush=True)
    try:
        msg = fn()
        print(f"\r[OK] {name}")
        if msg:
            print(indent(msg, "       "))
        return True
    except ImportError as e:
        print(f"\r[--] {name}  (未安装: {e.name})")
        return False
    except Exception as e:  # noqa: BLE001
        print(f"\r[FAIL] {name}")
        print(indent(f"  原因: {type(e).__name__}: {e}", "       "))
        return False


def check_python() -> str:
    return f"Python {platform.python_version()}  on  {platform.machine()}"


def check_pytorch() -> str:
    import torch

    info = [f"torch {torch.__version__}"]
    info.append(f"MPS 可用: {torch.backends.mps.is_available()}")
    info.append(f"MPS built: {torch.backends.mps.is_built()}")

    # 简单前向反向测试
    if torch.backends.mps.is_available():
        x = torch.randn(64, 64, device="mps", requires_grad=True)
        y = (x @ x.T).sum()
        y.backward()
        info.append("MPS 前向/反向 OK")
    return "\n".join(info)


def check_jax() -> str:
    import jax
    import jax.numpy as jnp

    info = [f"jax {jax.__version__}"]
    info.append(f"backend: {jax.default_backend()}")
    info.append(f"devices: {jax.devices()}")

    # 简单计算测试
    @jax.jit
    def f(x):
        return jnp.sum(x ** 2)

    val = float(f(jnp.arange(100.0)))
    info.append(f"jit 计算 OK (sample={val:.1f})")
    return "\n".join(info)


def check_mujoco() -> str:
    import mujoco

    info = [f"mujoco {mujoco.__version__}"]
    # 内置一个最小 XML
    xml = """
    <mujoco>
      <worldbody>
        <body name="b" pos="0 0 1">
          <freejoint/>
          <geom type="box" size=".1 .1 .1" rgba="1 0 0 1"/>
        </body>
      </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    for _ in range(100):
        mujoco.mj_step(model, data)
    info.append(f"模拟 100 步 OK，box 末端 z = {data.qpos[2]:.3f}")
    return "\n".join(info)


def check_mjx() -> str:
    import mujoco
    from mujoco import mjx

    xml = """
    <mujoco>
      <worldbody>
        <body name="b" pos="0 0 1"><freejoint/>
          <geom type="box" size=".1 .1 .1"/></body>
      </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    mjx_model = mjx.put_model(model)
    mjx_data = mjx.make_data(mjx_model)
    mjx_data = mjx.step(mjx_model, mjx_data)
    return f"MJX step OK, qpos.shape={mjx_data.qpos.shape}"


def check_gymnasium() -> str:
    import gymnasium as gym

    env = gym.make("CartPole-v1")
    obs, _ = env.reset(seed=0)
    env.step(env.action_space.sample())
    env.close()
    return f"gymnasium {gym.__version__}, CartPole-v1 OK"


def check_gym_mujoco() -> str:
    import gymnasium as gym

    env = gym.make("HalfCheetah-v4")
    obs, _ = env.reset(seed=0)
    env.step(env.action_space.sample())
    env.close()
    return "HalfCheetah-v4 OK（MuJoCo 集成正常）"


def check_sb3() -> str:
    import stable_baselines3 as sb3

    return f"stable-baselines3 {sb3.__version__}"


def check_robosuite() -> str:
    import robosuite as suite

    return f"robosuite {suite.__version__}"


def check_brax() -> str:
    import brax

    return f"brax {brax.__version__ if hasattr(brax, '__version__') else 'installed'}"


def check_playground() -> str:
    """MuJoCo Playground 检查。"""
    # 包名可能是 `playground` 也可能是 `mujoco_playground`，都试一下
    for name in ["mujoco_playground", "playground"]:
        try:
            mod = importlib.import_module(name)
            return f"{name} ({getattr(mod, '__version__', 'installed')})"
        except ImportError:
            continue
    raise ImportError("mujoco_playground")


def check_lerobot() -> str:
    import lerobot

    return f"lerobot ({getattr(lerobot, '__version__', 'installed')})"


# ----------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------

CHECKS = [
    ("Python 信息", check_python),
    ("PyTorch + MPS", check_pytorch),
    ("JAX + 后端", check_jax),
    ("MuJoCo", check_mujoco),
    ("MuJoCo MJX", check_mjx),
    ("Gymnasium", check_gymnasium),
    ("Gymnasium MuJoCo 任务", check_gym_mujoco),
    ("Stable-Baselines3", check_sb3),
    ("robosuite", check_robosuite),
    ("Brax", check_brax),
    ("MuJoCo Playground", check_playground),
    ("LeRobot", check_lerobot),
]


def main():
    print("=" * 64)
    print("  RL Robot Control - 环境健康检查")
    print("=" * 64)
    print()

    results = []
    for name, fn in CHECKS:
        ok = check(name, fn)
        results.append((name, ok))
        print()

    # 汇总
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    print("-" * 64)
    print(f"  通过: {passed}/{total}")
    print("-" * 64)

    # 给点提示
    failed = [name for name, ok in results if not ok]
    if not failed:
        print("\n所有检查通过。可以开始 week01_02_foundations 了。\n")
    else:
        print("\n未通过项：")
        for name in failed:
            print(f"  - {name}")
        print("\n如果是因为'未安装'，是预期的——一个环境不需要装所有库。")
        print("如果是 FAIL，请查看 setup/README.md 的'常见坑'。\n")
        sys.exit(1 if any("FAIL" in name for name in failed) else 0)


if __name__ == "__main__":
    main()
