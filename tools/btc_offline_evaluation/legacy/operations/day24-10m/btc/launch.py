"""Launch a CPU-only BTC session in its own named container; keep it alive at target."""
import hashlib,json,os,subprocess
from pathlib import Path
base=Path("/data0/REPLAY_USER/hit-rate-evaluation")
root=base/"day24-10m-20261106"
work=root/"btc"
out=work/"state"
out.mkdir(exist_ok=False)
(out/"authorized-through").write_text("1000\n")
exe=base/"btc-serial-20261103/source/bazel-bin/rtp_llm/cpp/cache/block_tree_cache/benchmark/resident_metadata_replay"
def digest(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for buf in iter(lambda:f.read(8<<20),b""):h.update(buf)
 return h.hexdigest()
image=subprocess.check_output(["docker","inspect","--format","{{.Image}}","REPLAY_USER_btc_v64_h110_20261013"],text=True).strip()
name="REPLAY_USER-btc-metadata-hold-10m-20261106"
args=["docker","run","-d","--name",name,"--network","none","--runtime=runc","--read-only","--cap-drop=ALL","--security-opt=no-new-privileges","--restart=no","--ulimit","core=0","--user",f"{os.getuid()}:{os.getgid()}","-e","CUDA_VISIBLE_DEVICES=","-e","NVIDIA_VISIBLE_DEVICES=void","-e","LOG_LEVEL=ERROR","-e","LD_LIBRARY_PATH=/driver:/opt/conda310/lib:/usr/local/cuda/lib64:/usr/lib64","-v",f"{base}:{base}:ro","-v",f"{out}:/out:rw","-w","/out"]
for lib in ["libcuda.so.1","libnvidia-ml.so.1"]:args+=["-v",f"{(Path('/lib64')/lib).resolve(strict=True)}:/driver/{lib}:ro"]
args+=["--entrypoint",str(exe),image,str(work/"finite-1to1-capacity.json"),str(root/"input"),"/out"]
(out/"launch.json").write_text(json.dumps(dict(image=image,name=name,binary_sha256=digest(exe),overlay_sha256=digest(work/"btc-10m-build.tgz"),input_sha256=digest(root/"input/verification.json"),argv=args,baseline_commit="08cf512139934301f732c98f7379a713ee88370d",swa="all reusable checkpoints",linear_step=1,capacity=json.loads((work/"finite-1to1-capacity.json").read_text())),indent=2))
print(subprocess.check_output(args,text=True).strip(),flush=True)
