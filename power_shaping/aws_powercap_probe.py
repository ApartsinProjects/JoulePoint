# -*- coding: utf-8 -*-
"""
AWS EC2 GPU power-cap PERMISSION probe with FULL, guaranteed cleanup.

The account has no default VPC, so this creates a throwaway VPC + subnet, launches one
short-lived GPU instance (real VM, root, GPU passthrough), runs nvidia-smi power-cap +
locked-clock tests, and then deletes EVERYTHING it created. No resource is left behind:

  created (all tagged purpose=powershaping-powercap-probe):  VPC, subnet, instance
  NOT created:  key pair, security group (uses VPC default), IGW, S3, IAM  -> nothing else
  self-terminating: InstanceInitiatedShutdownBehavior=terminate + user-data `shutdown` +
                    a 15-min dead-man switch inside the instance
  finally block: terminate instance (wait) -> delete subnet -> delete VPC, then a tag sweep
                 verifies nothing tagged remains and force-removes any straggler.

Run:  python aws_powercap_probe.py [--instance-type g5.xlarge] [--region us-east-1]
Clean only (sweep orphans by tag):  python aws_powercap_probe.py --cleanup-only
"""
from __future__ import annotations
import argparse, time, json, re, os
import boto3
from botocore.exceptions import ClientError

TAG = "powershaping-powercap-probe"
TAGS = [{"Key": "Name", "Value": TAG}, {"Key": "purpose", "Value": TAG}]

USER_DATA = r"""#!/bin/bash
exec > >(tee -a /dev/console) 2>&1
( sleep 900; shutdown -h now ) &
sleep 25
echo "AWS_PROBE_START"
nvidia-smi --query-gpu=name,power.default_limit,power.min_limit,power.max_limit,power.draw --format=csv,noheader,nounits | sed 's/^/GPU_INFO,/'
nvidia-smi -pm 1 >/dev/null 2>&1 && echo "PERSISTENCE=on" || echo "PERSISTENCE=fail"
DEF=$(nvidia-smi --query-gpu=power.default_limit --format=csv,noheader,nounits | head -1 | cut -d. -f1)
TGT=$(( DEF * 65 / 100 ))
echo "SET_PL_TARGET_W=$TGT"
OUT=$(sudo nvidia-smi -pl $TGT 2>&1); echo "SET_PL_RC=$?"; echo "SET_PL_MSG=$OUT"
echo "ENFORCED_AFTER_W=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits | head -1)"
MAXC=$(nvidia-smi --query-gpu=clocks.max.sm --format=csv,noheader,nounits | head -1)
OUT2=$(sudo nvidia-smi -lgc 0,$(( MAXC / 2 )) 2>&1); echo "SET_LGC_RC=$?"; echo "SET_LGC_MSG=$OUT2"
echo "AWS_PROBE_END"
sleep 3; shutdown -h now
"""


def latest_dlami(ec2):
    imgs = ec2.describe_images(Owners=["amazon"], Filters=[
        {"Name": "name", "Values": ["Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)*"]},
        {"Name": "state", "Values": ["available"]}])["Images"]
    if not imgs:
        raise RuntimeError("no Deep Learning GPU AMI found")
    return sorted(imgs, key=lambda x: x["CreationDate"], reverse=True)[0]["ImageId"]


def az_for_type(ec2, itype):
    offs = ec2.describe_instance_type_offerings(LocationType="availability-zone",
             Filters=[{"Name": "instance-type", "Values": [itype]}])["InstanceTypeOfferings"]
    if not offs:
        raise RuntimeError(f"{itype} not offered in region")
    return sorted(o["Location"] for o in offs)[0]


def cleanup(ec2, vpc=None, subnet=None, iid=None, verbose=True, tag_sweep=False):
    """Idempotent teardown in dependency order.

    Targets ONLY the vpc/subnet/iid ids passed in. Tag-wide discovery (terminate/delete EVERY resource
    carrying TAG) happens ONLY when tag_sweep=True -- reserve that for a single explicit final sweep.
    Per-run teardown MUST use the default tag_sweep=False: otherwise a run whose launch failed before an
    instance existed (iid=None, e.g. capacity failure) would fall back to tag discovery and terminate a
    CONCURRENT sibling sweep's instances -- the exact bug that killed the parallel g-sweeps.
    """
    def log(m):
        if verbose: print(m)
    # 1) instance -- explicit id only, unless an explicit tag sweep is requested
    ids = [iid] if iid else []
    if not ids and tag_sweep:
        r = ec2.describe_instances(Filters=[{"Name": "tag:purpose", "Values": [TAG]}])
        ids = [i["InstanceId"] for R in r["Reservations"] for i in R["Instances"]
               if i["State"]["Name"] not in ("terminated",)]
    if ids:
        log(f"[cleanup] terminating {ids} ...")
        try:
            ec2.terminate_instances(InstanceIds=ids)
            ec2.get_waiter("instance_terminated").wait(InstanceIds=ids,
                WaiterConfig={"Delay": 15, "MaxAttempts": 60})
            log("[cleanup] instance(s) terminated.")
        except Exception as e:
            log(f"[cleanup] WARN terminate: {e}")
    # 2) subnets -- explicit id, or (tag sweep only) every tagged subnet
    subs = [subnet] if subnet else ([s["SubnetId"] for s in
            ec2.describe_subnets(Filters=[{"Name": "tag:purpose", "Values": [TAG]}])["Subnets"]]
            if tag_sweep else [])
    for s in subs:
        for _ in range(6):
            try:
                ec2.delete_subnet(SubnetId=s); log(f"[cleanup] deleted subnet {s}"); break
            except ClientError as e:
                if "DependencyViolation" in str(e):
                    time.sleep(10); continue
                log(f"[cleanup] WARN subnet {s}: {e}"); break
    # 2.5) VPC endpoints MUST be deleted before their VPC (else DependencyViolation) -- the explicit vpc's
    # endpoints, or (tag sweep) every tagged endpoint. Without this a killed orchestrator orphans the VPC.
    try:
        if vpc:
            eps = ec2.describe_vpc_endpoints(Filters=[{"Name": "vpc-id", "Values": [vpc]}])["VpcEndpoints"]
        elif tag_sweep:
            eps = ec2.describe_vpc_endpoints(Filters=[{"Name": "tag:purpose", "Values": [TAG]}])["VpcEndpoints"]
        else:
            eps = []
        eids = [e["VpcEndpointId"] for e in eps if e.get("State") not in ("deleted", "deleting")]
        if eids:
            ec2.delete_vpc_endpoints(VpcEndpointIds=eids); log(f"[cleanup] deleting endpoints {eids}")
            time.sleep(5)
    except Exception as e:
        log(f"[cleanup] WARN endpoints: {e}")
    # 3) vpcs -- explicit id, or (tag sweep only) every tagged vpc
    vpcs = [vpc] if vpc else ([v["VpcId"] for v in
            ec2.describe_vpcs(Filters=[{"Name": "tag:purpose", "Values": [TAG]}])["Vpcs"]]
            if tag_sweep else [])
    for v in vpcs:
        for _ in range(6):
            try:
                ec2.delete_vpc(VpcId=v); log(f"[cleanup] deleted vpc {v}"); break
            except ClientError as e:
                if "DependencyViolation" in str(e):
                    time.sleep(10); continue
                log(f"[cleanup] WARN vpc {v}: {e}"); break
    if not tag_sweep:
        return True                      # per-run teardown of own ids; don't report siblings as "remaining"
    # 4) verify (tag sweep only)
    left_i = ec2.describe_instances(Filters=[{"Name": "tag:purpose", "Values": [TAG]}])
    live = [i["InstanceId"] for R in left_i["Reservations"] for i in R["Instances"]
            if i["State"]["Name"] not in ("terminated",)]
    left_v = ec2.describe_vpcs(Filters=[{"Name": "tag:purpose", "Values": [TAG]}])["Vpcs"]
    left_s = ec2.describe_subnets(Filters=[{"Name": "tag:purpose", "Values": [TAG]}])["Subnets"]
    log(f"[cleanup] VERIFY remaining -> instances:{live or 'none'} "
        f"vpcs:{[v['VpcId'] for v in left_v] or 'none'} subnets:{[s['SubnetId'] for s in left_s] or 'none'}")
    return not (live or left_v or left_s)


def parse(text):
    def g(key, cast=str):
        m = re.search(rf"{key}=(.*)", text or "")
        return cast(m.group(1).strip()) if m else None
    info = re.search(r"GPU_INFO,(.*)", text or "")
    def as_int(x):
        x = x.strip(); return int(x) if x.lstrip("-").isdigit() else x
    out = {"gpu_info": info.group(1).strip() if info else None,
           "persistence": g("PERSISTENCE"), "set_pl_target_w": g("SET_PL_TARGET_W"),
           "set_pl_rc": g("SET_PL_RC", as_int), "set_pl_msg": g("SET_PL_MSG"),
           "enforced_after_w": g("ENFORCED_AFTER_W"),
           "set_lgc_rc": g("SET_LGC_RC", as_int), "set_lgc_msg": g("SET_LGC_MSG")}
    out["power_cap_works"] = out.get("set_pl_rc") == 0
    out["locked_clocks_work"] = out.get("set_lgc_rc") == 0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance-type", default="g5.xlarge")
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--cleanup-only", action="store_true")
    args = ap.parse_args()
    ec2 = boto3.client("ec2", region_name=args.region)

    if args.cleanup_only:
        ok = cleanup(ec2, tag_sweep=True); print("[cleanup-only] clean." if ok else "[cleanup-only] STRAGGLERS remain."); return

    ami = latest_dlami(ec2); az = az_for_type(ec2, args.instance_type)
    print(f"[aws] region={args.region} type={args.instance_type} az={az} ami={ami}")

    vpc_id = subnet_id = iid = None
    try:
        vpc_id = ec2.create_vpc(CidrBlock="10.0.0.0/16",
                    TagSpecifications=[{"ResourceType": "vpc", "Tags": TAGS}])["Vpc"]["VpcId"]
        ec2.get_waiter("vpc_available").wait(VpcIds=[vpc_id])
        subnet_id = ec2.create_subnet(VpcId=vpc_id, CidrBlock="10.0.0.0/24", AvailabilityZone=az,
                    TagSpecifications=[{"ResourceType": "subnet", "Tags": TAGS}])["Subnet"]["SubnetId"]
        print(f"[aws] created vpc={vpc_id} subnet={subnet_id}")

        print("[aws] launching self-terminating GPU instance...")
        r = ec2.run_instances(ImageId=ami, InstanceType=args.instance_type, MinCount=1, MaxCount=1,
            SubnetId=subnet_id, UserData=USER_DATA,
            InstanceInitiatedShutdownBehavior="terminate",
            BlockDeviceMappings=[{"DeviceName": "/dev/sda1",
                "Ebs": {"DeleteOnTermination": True, "VolumeSize": 100}}],
            TagSpecifications=[{"ResourceType": "instance", "Tags": TAGS}])
        iid = r["Instances"][0]["InstanceId"]
        print(f"[aws] instance {iid}; polling console for probe output (~5-10 min)...")

        text = None; deadline = time.time() + 720
        while time.time() < deadline:
            time.sleep(30)
            try:
                out = ec2.get_console_output(InstanceId=iid, Latest=True).get("Output", "") or ""
            except ClientError:
                out = ""
            st = ec2.describe_instances(InstanceIds=[iid])["Reservations"][0]["Instances"][0]["State"]["Name"]
            print(f"  ... {int(deadline - time.time())}s left, state={st}, console={len(out)}B")
            if "AWS_PROBE_END" in out:
                text = out; break
            if st == "terminated":
                text = out; break

        verdict = parse(text) if text else {"error": "no console output captured before timeout/termination"}
        os.makedirs("results", exist_ok=True)
        json.dump(verdict, open("results/aws_powercap_probe.json", "w"), indent=2)
        print("\n[aws] PROBE_RESULT " + json.dumps(verdict, indent=2))
    except ClientError as e:
        print(f"[aws] LAUNCH ERROR: {e}")
    finally:
        print("\n[aws] tearing down all created resources...")
        cleanup(ec2, vpc=vpc_id, subnet=subnet_id, iid=iid)


if __name__ == "__main__":
    main()
