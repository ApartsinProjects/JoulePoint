# -*- coding: utf-8 -*-
"""
EXPANDED AWS power-cap elasticity sweep -- the data-collection campaign the model-improvement
diagnosis called for (more DISTINCT, DIVERSE workloads at a consistent construct).

Runs ~22 workloads spanning the arithmetic-intensity spectrum (compute-bound GEMM/conv/attention
-> memory-bound elementwise/gather/reduction -> latency-bound decode) at 6 power caps, on one or
more real GPUs (default: A10G g5, L4 g6, T4 g4dn). For each GPU it measures NVML power +
throughput per (workload, cap).

Outputs:
  * data/raw/own_aws_sweep.csv     -- the A10G run, same schema the model loaders already read,
                                      so the within-construct workload count jumps from ~7 to ~22.
  * data/raw/aws_sweep_multi.csv   -- every GPU run, with a `gpu` column, for cross-hardware.

Self-cleaning: throwaway VPC/subnet, self-terminating instance, AZ-retry on capacity, per-GPU
try/finally teardown, and a final tag-based orphan sweep. Nothing is left running.
"""
from __future__ import annotations
import argparse, time, re, os, base64, io, csv, gzip, uuid
import boto3
from botocore.exceptions import ClientError
from aws_powercap_probe import cleanup, TAG, TAGS
from aws_powercap_sweep import latest_pytorch_dlami

ACCOUNT = "388342378766"


def ensure_bucket(region):
    """A per-region results bucket (a gateway VPC endpoint reaches S3 only in the instance's own region,
    so the bucket must live there). Returns (bucket_name, s3_client)."""
    b = f"psweeps-{ACCOUNT}-{region}"
    s3 = boto3.client("s3", region_name=region)
    try:
        s3.head_bucket(Bucket=b)
    except ClientError:
        if region == "us-east-1":
            s3.create_bucket(Bucket=b)
        else:
            s3.create_bucket(Bucket=b, CreateBucketConfiguration={"LocationConstraint": region})
    return b, s3

# ---------------- on-instance sweep program (expanded workload set) ----------------
SWEEP_PY = r'''
import time, subprocess, threading, sys
import torch

def smi(q):
    return subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
                          capture_output=True, text=True).stdout.strip().splitlines()

subprocess.run(["nvidia-smi", "-pm", "1"], capture_output=True)
name = smi("name")[0].replace(",", " ")
defW = int(float(smi("power.default_limit")[0])); minW = int(float(smi("power.min_limit")[0]))
print(f"SWEEP_GPU={name} default={defW} min={minW}", flush=True)

try:
    import pynvml; pynvml.nvmlInit(); _h = pynvml.nvmlDeviceGetHandleByIndex(0)
    def draw(): return pynvml.nvmlDeviceGetPowerUsage(_h)/1000.0
except Exception:
    def draw():
        try: return float(smi("power.draw")[0])
        except Exception: return float("nan")

class Sampler(threading.Thread):
    def __init__(s): super().__init__(daemon=True); s.on=True; s.v=[]
    def run(s):
        while s.on:
            s.v.append(draw()); time.sleep(0.05)
    def stop(s): s.on=False; return s.v

dev = torch.device("cuda"); torch.backends.cudnn.benchmark = True
import torchvision.models as tvm

def gemm(dtype, n=8192):
    a = torch.randn(n, n, device=dev, dtype=dtype); b = torch.randn(n, n, device=dev, dtype=dtype)
    def step(): return (a @ b).sum()
    return step
def batched_matmul(dtype):
    a = torch.randn(256, 512, 512, device=dev, dtype=dtype); b = torch.randn(256, 512, 512, device=dev, dtype=dtype)
    def step(): return torch.bmm(a, b).sum()
    return step
def attention():
    q = torch.randn(32, 16, 512, 64, device=dev, dtype=torch.float16)
    k = torch.randn(32, 16, 512, 64, device=dev, dtype=torch.float16)
    v = torch.randn(32, 16, 512, 64, device=dev, dtype=torch.float16)
    def step():
        with torch.no_grad():
            return torch.nn.functional.scaled_dot_product_attention(q, k, v).sum()
    return step
def vision(builder, bs=32):
    m = builder(weights=None).to(dev).half().eval()
    x = torch.randn(bs, 3, 224, 224, device=dev, dtype=torch.float16)
    def step():
        with torch.no_grad(): return m(x).sum()
    return step
def fft_wl():
    x = torch.randn(2048, 2048, device=dev, dtype=torch.float32)
    def step(): return torch.fft.fft2(x).real.sum()
    return step
def cholesky_wl():
    a = torch.randn(2048, 2048, device=dev, dtype=torch.float32); a = a @ a.t() + 2048*torch.eye(2048, device=dev)
    def step(): return torch.linalg.cholesky(a).sum()
    return step
def membw():
    n = 1<<26; x = torch.randn(n, device=dev); y = torch.randn(n, device=dev)
    def step(): return (x*1.0001 + y)[0]
    return step
def reduction():
    x = torch.randn(1<<26, device=dev)
    def step(): return x.sum()
    return step
def softmax_big():
    x = torch.randn(4096, 16384, device=dev, dtype=torch.float16)
    def step(): return torch.softmax(x, dim=-1).sum()
    return step
def layernorm_big():
    x = torch.randn(8192, 8192, device=dev, dtype=torch.float16); ln = torch.nn.LayerNorm(8192).half().to(dev)
    def step():
        with torch.no_grad(): return ln(x).sum()
    return step
def embed_gather():
    emb = torch.nn.Embedding(4_000_000, 128).to(dev); idx = torch.randint(0, 4_000_000, (65536,), device=dev)
    def step(): return emb(idx).sum()
    return step
def scatter_add():
    n = 1<<24; src = torch.randn(n, device=dev); idx = torch.randint(0, n, (n,), device=dev); out = torch.zeros(n, device=dev)
    def step(): return out.scatter_add(0, idx, src).sum()
    return step
def sort_big():
    x = torch.randn(1<<23, device=dev)
    def step(): return torch.sort(x)[0][0]
    return step
def memcpy():
    x = torch.randn(1<<26, device=dev)
    def step(): return x.clone()[0]
    return step
def decode_like():
    a = torch.randn(512, 4096, device=dev, dtype=torch.float16); b = torch.randn(4096, 4096, device=dev, dtype=torch.float16)
    def step(): c = a @ b; torch.cuda.synchronize(); return c.sum()
    return step
def elementwise_chain():
    x = torch.randn(1<<24, device=dev)
    def step():
        y = x
        for _ in range(8): y = torch.sin(y) + 0.5*y
        return y[0]
    return step

BUILDERS = {
    "gemm_fp16": lambda: gemm(torch.float16), "gemm_fp32": lambda: gemm(torch.float32),
    "gemm_bf16": lambda: gemm(torch.bfloat16), "bmm_fp16": lambda: batched_matmul(torch.float16),
    "attention_sdpa": attention, "fft2d": fft_wl, "cholesky": cholesky_wl,
    "resnet50": lambda: vision(tvm.resnet50), "resnet152": lambda: vision(tvm.resnet152),
    "vgg16": lambda: vision(tvm.vgg16), "densenet121": lambda: vision(tvm.densenet121),
    "convnext_tiny": lambda: vision(tvm.convnext_tiny), "efficientnet_b0": lambda: vision(tvm.efficientnet_b0),
    "mobilenet_v3_l": lambda: vision(tvm.mobilenet_v3_large), "vit_b16": lambda: vision(tvm.vit_b_16),
    "inception_v3": lambda: vision(tvm.inception_v3),
    "membw": membw, "reduction": reduction, "softmax_big": softmax_big, "layernorm_big": layernorm_big,
    "embed_gather": embed_gather, "scatter_add": scatter_add, "sort_big": sort_big, "memcpy": memcpy,
    "decode_like": decode_like, "elementwise_chain": elementwise_chain,
}

CAPS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
REPS = 2
def set_cap(fr):
    w = max(minW, int(round(fr*defW)))
    subprocess.run(["sudo","nvidia-smi","-pl",str(w)], capture_output=True); return w

def measure(step, secs=3.5):
    torch.cuda.synchronize(); s = Sampler(); s.start()
    t0 = time.time(); n = 0
    while time.time()-t0 < secs:
        step(); n += 1
    torch.cuda.synchronize(); el = time.time()-t0
    v = [x for x in s.stop() if x==x]
    p = sum(v)/len(v) if v else float("nan")
    return n/el, p

print("SWEEP_CSV_START", flush=True)
print("gpu,workload,cap_frac,cap_w,rep,power_w,throughput", flush=True)
for wname, builder in BUILDERS.items():
    try:
        step = builder()
        for _ in range(2): step()
        torch.cuda.synchronize()
    except Exception as e:
        print(f"# {wname} build/warm failed: {str(e)[:70]}", flush=True); continue
    for fr in CAPS:
        w = set_cap(fr); time.sleep(1.5)
        for r in range(REPS):
            try:
                thr, pw = measure(step)
                print(f"{name},{wname},{fr:.2f},{w},{r},{pw:.1f},{thr:.5f}", flush=True)
            except Exception as e:
                print(f"# {wname},{fr},{w},{r} err {str(e)[:50]}", flush=True)
    set_cap(1.0)
    del step
    torch.cuda.empty_cache()
print("SWEEP_CSV_END", flush=True)
'''

USER_DATA_TMPL = r"""#!/bin/bash
# REQUIRED: force ALL script output to the serial console at any time. Cloud-init only echoes BOOT-PHASE
# output (~first 2 min), so a blob emitted 30 min into the run is otherwise LOST from get_console_output.
# The double-echo this causes is handled by blob-last (blob stays in the 64KB tail) + chunk de-dup.
exec > >(tee -a /dev/console) 2>&1
( sleep 3300; shutdown -h now ) &      # 55-min dead-man switch (room for ~20 full-grid models)
sleep 25
echo "SWEEP_BOOT_OK"
echo "{B64}" | base64 -d > /root/sweep.py
source /opt/conda/etc/profile.d/conda.sh 2>/dev/null && conda activate pytorch 2>/dev/null
PY=python3
for cand in "$(which python)" /opt/conda/envs/pytorch/bin/python /opt/conda/bin/python /opt/pytorch/bin/python /usr/bin/python3; do
  if [ -n "$cand" ] && $cand -c "import torch, torchvision" 2>/dev/null; then PY=$cand; break; fi
done
echo "SWEEP_PYTHON=$PY"
$PY /root/sweep.py > /root/out.csv 2>/root/err.log; echo "SWEEP_PY_EXIT=$?"
# PRIMARY data path: upload the results file to S3 via a presigned PUT (no size limit, no console fragility).
gzip -c /root/out.csv > /root/out.csv.gz
if [ "{PUT_URL}" != NONE ]; then
  for a in 1 2 3 4 5; do
    if curl -sSf --connect-timeout 15 --max-time 300 -X PUT --upload-file /root/out.csv.gz "{PUT_URL}"; then echo "SWEEP_UPLOADED bytes=$(stat -c%s /root/out.csv.gz)"; break; else echo "SWEEP_UPLOAD_RETRY=$a"; sleep 5; fi
  done
fi
# FALLBACK data path: gzip+base64 blob on the serial console (tee forces late output to the console; blob-last
# keeps it in the 64KB tail; chunks de-duped by the parser). Used only if the S3 object is missing.
echo "SWEEP_ERR_HEAD"; head -c 1500 /root/err.log; echo
echo "SWEEP_B64_START"
gzip -c /root/out.csv | base64 -w 76 | awk '{printf "B64[%05d]%s\n", NR, $0}'
echo "SWEEP_B64_END"
sleep 3; shutdown -h now
"""


def run_one(ec2, itype, ami, user_data, bucket=None, s3=None, s3_prefix="sweeps"):
    """Launch one GPU sweep with AZ-retry; return (rows, gpu_name). Always tears down.

    If bucket/s3 are given, the instance uploads its results file to S3 via a presigned PUT (PRIMARY path),
    reached through a free S3 GATEWAY endpoint on the throwaway VPC -- no size limit, no console fragility.
    The serial-console gzip blob remains an automatic fallback.
    """
    region = ec2.meta.region_name
    azs = sorted(o["Location"] for o in ec2.describe_instance_type_offerings(
        LocationType="availability-zone",
        Filters=[{"Name": "instance-type", "Values": [itype]}])["InstanceTypeOfferings"])
    if not azs:
        print(f"[aws] {itype}: not offered in this region, skipping"); return [], None
    key = None
    if bucket and s3:
        key = f"{s3_prefix}/{itype}-{uuid.uuid4().hex[:10]}.csv.gz"
        # Sign against the REGIONAL endpoint (host = s3.<region>.amazonaws.com), which the S3 gateway endpoint
        # routes. The default global host (bucket.s3.amazonaws.com) is NOT covered outside us-east-1 -> the
        # instance has no route and curl times out (the exact Tokyo failure).
        s3_sign = boto3.client("s3", region_name=region, endpoint_url=f"https://s3.{region}.amazonaws.com")
        put_url = s3_sign.generate_presigned_url("put_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=10800)
        user_data = user_data.replace("{PUT_URL}", put_url)
    else:
        user_data = user_data.replace("{PUT_URL}", "NONE")
    vpc_id = subnet_id = iid = eid = None
    try:
        vpc_id = ec2.create_vpc(CidrBlock="10.0.0.0/16",
                    TagSpecifications=[{"ResourceType": "vpc", "Tags": TAGS}])["Vpc"]["VpcId"]
        ec2.get_waiter("vpc_available").wait(VpcIds=[vpc_id])
        if bucket:                                    # S3 gateway endpoint on the VPC's main route table
            rts = ec2.describe_route_tables(Filters=[{"Name": "vpc-id", "Values": [vpc_id]}])["RouteTables"]
            main_rt = next((t["RouteTableId"] for t in rts for a in t.get("Associations", []) if a.get("Main")),
                           rts[0]["RouteTableId"])
            eid = ec2.create_vpc_endpoint(VpcEndpointType="Gateway", VpcId=vpc_id,
                    ServiceName=f"com.amazonaws.{region}.s3", RouteTableIds=[main_rt],
                    TagSpecifications=[{"ResourceType": "vpc-endpoint", "Tags": TAGS}])["VpcEndpoint"]["VpcEndpointId"]
        for i, az in enumerate(azs):
            subnet_id = ec2.create_subnet(VpcId=vpc_id, CidrBlock=f"10.0.{i}.0/24", AvailabilityZone=az,
                        TagSpecifications=[{"ResourceType": "subnet", "Tags": TAGS}])["Subnet"]["SubnetId"]
            try:
                r = ec2.run_instances(ImageId=ami, InstanceType=itype, MinCount=1, MaxCount=1,
                    SubnetId=subnet_id, UserData=user_data, InstanceInitiatedShutdownBehavior="terminate",
                    BlockDeviceMappings=[{"DeviceName": "/dev/sda1", "Ebs": {"DeleteOnTermination": True, "VolumeSize": 100}}],
                    TagSpecifications=[{"ResourceType": "instance", "Tags": TAGS}])
                iid = r["Instances"][0]["InstanceId"]; print(f"[aws] {itype} launched in {az} ({iid})"); break
            except ClientError as e:
                if "InsufficientInstanceCapacity" in str(e) and i < len(azs) - 1:
                    print(f"[aws] {itype}: no capacity in {az}, next AZ")
                    ec2.delete_subnet(SubnetId=subnet_id); subnet_id = None; continue
                raise
        if iid is None:
            print(f"[aws] {itype}: no capacity in any AZ, skipping"); return [], None

        text = None; deadline = time.time() + 3000
        while time.time() < deadline:
            time.sleep(45)
            try:
                out = ec2.get_console_output(InstanceId=iid, Latest=True).get("Output", "") or ""
            except ClientError:
                out = ""
            st = ec2.describe_instances(InstanceIds=[iid])["Reservations"][0]["Instances"][0]["State"]["Name"]
            got = ("SWEEP_B64_END" in out) or ("SWEEP_CSV_END" in out)
            print(f"  ... {itype} {int(deadline-time.time())}s left, state={st}, console={len(out)}B, done={got}")
            if out:
                text = out
            if got or st in ("shutting-down", "terminated"):
                break

        rows = []
        if bucket and s3 and key:                       # PRIMARY: download the results file from S3
            for _ in range(8):
                try:
                    obj = s3.get_object(Bucket=bucket, Key=key)
                    csvtext = gzip.decompress(obj["Body"].read()).decode("utf-8", "replace")
                    for d in csv.DictReader(io.StringIO(csvtext)):
                        wl = d.get("workload")
                        if not wl or wl.startswith("#") or (d.get("gpu") or "").lstrip().startswith("#") or wl == "workload" or not d.get("power_w"):
                            continue
                        rows.append(d)
                    print(f"[aws] {itype}: downloaded {len(rows)} rows from s3://{bucket}/{key}")
                    try:
                        s3.delete_object(Bucket=bucket, Key=key)
                    except Exception:
                        pass
                    break
                except s3.exceptions.NoSuchKey:
                    time.sleep(12)                       # upload happens near shutdown; give it a moment
                except Exception as e:
                    print(f"[aws] {itype}: S3 get warn: {type(e).__name__}: {str(e)[:50]}"); time.sleep(12)
        if not rows and text and "SWEEP_B64_START" in text:
            # FALLBACK gzip+base64 blob path (robust to the 64KB console cap): reassemble indexed chunks,
            # de-duping the serial console's double-echo by chunk index, then gunzip -> CSV.
            seg = text.split("SWEEP_B64_START", 1)[1].split("SWEEP_B64_END", 1)[0]
            seg = re.sub(r"(?m)^\[[^\]]*\]\s*cloud-init\[\d+\]:\s*", "", seg)
            chunks = {}
            for m in re.finditer(r"B64\[(\d+)\]([A-Za-z0-9+/=]+)", seg):
                chunks[int(m.group(1))] = m.group(2)
            try:
                b64 = "".join(chunks[k] for k in sorted(chunks))
                csvtext = gzip.decompress(base64.b64decode(b64)).decode("utf-8", "replace")
                for d in csv.DictReader(io.StringIO(csvtext)):
                    wl = d.get("workload")
                    if not wl or wl.startswith("#") or (d.get("gpu") or "").lstrip().startswith("#") or wl == "workload" or not d.get("power_w"):
                        continue
                    rows.append(d)
                print(f"[aws] {itype}: decoded {len(rows)} rows from gzip blob ({len(chunks)} chunks)")
                if not rows:
                    print(f"[aws] {itype}: 0 data rows -- decoded content (diagnosis):\n{csvtext[:2500]}")
            except Exception as e:
                print(f"[aws] {itype}: blob decode FAILED: {type(e).__name__}: {e}")
        elif not rows and text and "SWEEP_CSV_START" in text:
            block = text.split("SWEEP_CSV_START", 1)[1].split("SWEEP_CSV_END", 1)[0]
            block = re.sub(r"(?m)^\[[^\]]*\]\s*cloud-init\[\d+\]:\s*", "", block)
            # the serial console echoes each line twice (tee + cloud-init), so drop the header
            # echo and dedup identical measurement lines.
            seen = set()
            for d in csv.DictReader(io.StringIO(block.strip().replace("\r", ""))):
                wl = d.get("workload")
                if not wl or wl.startswith("#") or (d.get("gpu") or "").lstrip().startswith("#") or wl == "workload" or not d.get("power_w"):
                    continue
                key = (wl, d.get("cap_frac"), d.get("rep"), d.get("power_w"), d.get("throughput"))
                if key in seen:
                    continue
                seen.add(key); rows.append(d)
        if rows and rows[0].get("gpu"):
            gpu_name = rows[0]["gpu"].replace("NVIDIA ", "").replace("Tesla ", "").strip()
        else:
            gpu = re.search(r"SWEEP_GPU=(.*?) default", text or "")
            gpu_name = gpu.group(1).strip() if gpu else itype
        print(f"[aws] {itype}: captured {len(rows)} rows on {gpu_name}")
        if not rows:
            sw = [l for l in (text or "").splitlines() if ("SWEEP_" in l) or l.lstrip().startswith("B64[")]
            print(f"[aws] {itype} NO CSV. console={len(text or '')}B. SWEEP markers ({len(sw)}):\n"
                  + "\n".join(sw[:15]) + ("\n...\n" + "\n".join(sw[-5:]) if len(sw) > 20 else "")
                  + f"\n--- tail ---\n{(text or '')[-700:]}")
        return rows, gpu_name
    finally:
        print(f"[aws] tearing down {itype} ...")
        if eid:                                          # endpoint must go before the VPC delete
            try:
                ec2.delete_vpc_endpoints(VpcEndpointIds=[eid])
            except Exception as e:
                print(f"[aws] {itype}: endpoint del warn: {str(e)[:50]}")
        cleanup(ec2, vpc=vpc_id, subnet=subnet_id, iid=iid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance-types", default="g5.xlarge,g6.xlarge,g4dn.xlarge",
                    help="comma-separated; g5=A10G, g6=L4, g4dn=T4")
    ap.add_argument("--region", default="us-east-1")
    args = ap.parse_args()
    ec2 = boto3.client("ec2", region_name=args.region)
    ami = latest_pytorch_dlami(ec2)
    user_data = USER_DATA_TMPL.replace("{B64}", base64.b64encode(SWEEP_PY.encode()).decode())
    types = [t.strip() for t in args.instance_types.split(",") if t.strip()]
    print(f"[aws] expanded sweep ami={ami} types={types}")

    os.makedirs("data/raw", exist_ok=True)
    multi_path = "data/raw/aws_sweep_multi.csv"
    multi_fields = ["gpu", "workload", "cap_frac", "cap_w", "rep", "power_w", "throughput"]
    all_rows = []
    try:
        for itype in types:
            rows, gpu_name = run_one(ec2, itype, ami, user_data)
            for d in rows:
                d.setdefault("gpu", gpu_name)
            all_rows += rows
            # write A10G (g5) rows to own_aws_sweep.csv (the model loaders' schema, no gpu col)
            if itype.startswith("g5.") and rows:
                with open("data/raw/own_aws_sweep.csv", "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=["workload", "cap_frac", "cap_w", "rep", "power_w", "throughput"])
                    w.writeheader()
                    for d in rows:
                        w.writerow({k: d[k] for k in ["workload", "cap_frac", "cap_w", "rep", "power_w", "throughput"]})
                print(f"[aws] wrote {len(rows)} A10G rows -> data/raw/own_aws_sweep.csv")
            # append everything to the multi-GPU file after each run (crash-safe)
            if all_rows:
                with open(multi_path, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=multi_fields, extrasaction="ignore")
                    w.writeheader(); [w.writerow(d) for d in all_rows]
        nwl = len({d["workload"] for d in all_rows}); ngpu = len({d["gpu"] for d in all_rows})
        print(f"\n[aws] TOTAL {len(all_rows)} rows, {nwl} distinct workloads, {ngpu} GPUs "
              f"-> {multi_path}")
    finally:
        print("\n[aws] final orphan sweep ...")
        ok = cleanup(ec2, tag_sweep=True)
        print("[aws] all clear." if ok else "[aws] WARNING: stragglers remain -- check console!")


if __name__ == "__main__":
    main()
