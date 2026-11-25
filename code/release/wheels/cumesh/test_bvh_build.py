"""cuBVH's tree build on several threads against one thread: compiles third_party/cubvh/src/bvh.cu of the fork
clone (--src) with test_bvh_build.cu for one architecture (no wheel needed, a minute) and runs it: random meshes of
4 to 2 M triangles built both ways and compared node by node and triangle by triangle (CUBVH_BUILD_CHECK), then the
time of a 4 M-triangle build with 1, 4, 16 and all threads.

    python test_bvh_build.py [--src C:/_myDrive/repos/CuMesh] [--eigen <an Eigen checkout>]
"""
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, r"C:\_myDrive\repos\spellcast3d\code\release\wheels\cumesh")
import build_cumesh_win as b  # noqa: E402

here = Path(__file__).resolve().parent
env = b.vcvars_env(b.DEFAULTS["vcvars"])
cuda = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8")
env["CUDA_PATH"] = env["CUDA_HOME"] = str(cuda)
src = Path(r"C:/_myDrive/repos/CuMesh\third_party\cubvh")
eigen = Path(r"C:\_myDrive\repos\IgorAherne\CuMesh\third_party\cubvh\third_party\eigen")
exe = Path(os.environ.get("TEMP", str(here))) / "test_bvh_build.exe"
flags = ["-O3", "-std=c++17", "--expt-relaxed-constexpr", "--extended-lambda", "-Xcompiler=/std:c++17",
         "-Xcompiler=/EHsc", "-Xcompiler=/permissive-", "-Xcompiler=/Zc:__cplusplus", "-allow-unsupported-compiler",
         "-gencode=arch=compute_86,code=sm_86", "-diag-suppress", "20208,177"]
cmd = [str(cuda / "bin" / "nvcc.exe"), str(src / "src" / "bvh.cu"), str(here / "test_bvh_build.cu"), "-o", str(exe),
       "-I", str(src / "include"), "-I", str(eigen)] + flags
t = time.perf_counter()
cp = subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=str(here))
print(f"compile exit {cp.returncode} in {time.perf_counter() - t:.0f} s")
errors = [l for l in (cp.stdout + cp.stderr).splitlines() if "error" in l.lower()]
print("\n".join(errors[:20]))
if cp.returncode == 0:
    run = subprocess.run([str(exe)], capture_output=True, text=True, cwd=str(here))
    print(run.stdout); print(run.stderr[-2000:]); print("run exit", run.returncode)
