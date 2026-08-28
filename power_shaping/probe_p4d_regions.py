# -*- coding: utf-8 -*-
"""
Probe p4d.24xlarge (8x A100) CAPACITY across regions, cheaply, BEFORE committing to a full run.

There is no free "capacity" API: DryRun only checks permissions, describe-instance-type-offerings only tells
you a type is OFFERED (not that a machine is free now). The only real signal is to attempt a launch and read
the result. So per (region, AZ) this creates a throwaway VPC/subnet, calls run_instances, and:
  * InsufficientInstanceCapacity -> no capacity in that AZ, try next;
  * success -> capacity EXISTS; terminate the instance IMMEDIATELY (pending->terminated bills ~$0);
  * VcpuLimitExceeded -> quota=0 in that region.

Isolation: a UNIQUE tag (p4d-quota-probe), NOT the sweep tag, and teardown ONLY by explicit id -- so it
cannot disturb a concurrent sweep. First filters to AZs that actually OFFER p4d (free) to avoid wasted work.
"""
import boto3
from aws_powercap_sweep import latest_pytorch_dlami

ITYPE = "p4d.24xlarge"
REGIONS = ["us-east-1", "us-east-2", "us-west-2", "eu-west-1", "eu-central-1", "ap-northeast-1"]
PTAGS = [{"Key": "Name", "Value": "p4d-quota-probe"}, {"Key": "purpose", "Value": "p4d-quota-probe"}]


def offered_azs(ec2):
    try:
        o = ec2.describe_instance_type_offerings(LocationType="availability-zone",
            Filters=[{"Name": "instance-type", "Values": [ITYPE]}])["InstanceTypeOfferings"]
        return sorted(x["Location"] for x in o)
    except Exception:
        return []


SPOT_OPTS = {"MarketType": "spot",
             "SpotOptions": {"SpotInstanceType": "one-time", "InstanceInterruptionBehavior": "terminate"}}


def _try_launch(ec2, ami, subnet, market):
    """market='ondemand'|'spot'. Returns ('capacity', iid) | ('nocap', None) | ('quota', None) | ('err:...', None)."""
    kw = dict(ImageId=ami, InstanceType=ITYPE, MinCount=1, MaxCount=1, SubnetId=subnet,
              InstanceInitiatedShutdownBehavior="terminate",
              TagSpecifications=[{"ResourceType": "instance", "Tags": PTAGS}])
    if market == "spot":
        kw["InstanceMarketOptions"] = SPOT_OPTS
    try:
        r = ec2.run_instances(**kw)
        iid = r["Instances"][0]["InstanceId"]
        ec2.terminate_instances(InstanceIds=[iid])                 # immediate: pending->terminated ~= $0
        return "capacity", iid
    except Exception as e:
        s = str(e)
        if "InsufficientInstanceCapacity" in s or "capacity-not-available" in s:
            return "nocap", None
        if "VcpuLimitExceeded" in s or "InstanceLimitExceeded" in s or "MaxSpotInstanceCountExceeded" in s:
            return "quota", None
        return f"err:{type(e).__name__}:{s[:40]}", None


def probe_region(region):
    ec2 = boto3.client("ec2", region_name=region)
    azs = offered_azs(ec2)
    if not azs:
        return f"{region:15} -> NOT OFFERED"
    try:
        ami = latest_pytorch_dlami(ec2)
    except Exception as e:
        return f"{region:15} -> no DLAMI ({str(e)[:30]})   [offered in {len(azs)} AZ]"
    vpc = subnet = None
    iids = []
    found = {}                                                     # market -> az
    quota = {}
    try:
        vpc = ec2.create_vpc(CidrBlock="10.9.0.0/16",
                             TagSpecifications=[{"ResourceType": "vpc", "Tags": PTAGS}])["Vpc"]["VpcId"]
        ec2.get_waiter("vpc_available").wait(VpcIds=[vpc])
        for i, az in enumerate(azs):
            try:
                subnet = ec2.create_subnet(VpcId=vpc, CidrBlock=f"10.9.{i}.0/24", AvailabilityZone=az,
                    TagSpecifications=[{"ResourceType": "subnet", "Tags": PTAGS}])["Subnet"]["SubnetId"]
            except Exception:
                subnet = None; continue
            for market in ("ondemand", "spot"):
                if market in found:
                    continue
                st, iid = _try_launch(ec2, ami, subnet, market)
                if iid:
                    iids.append(iid)
                if st == "capacity":
                    found[market] = az
                elif st == "quota":
                    quota[market] = True
            ec2.delete_subnet(SubnetId=subnet); subnet = None
            if len(found) == 2:
                break
    except Exception as e:
        return f"{region:15} -> setup ERR {type(e).__name__}: {str(e)[:40]}"
    finally:
        try:
            if iids:
                ec2.get_waiter("instance_terminated").wait(InstanceIds=iids,
                    WaiterConfig={"Delay": 10, "MaxAttempts": 30})
            if subnet:
                ec2.delete_subnet(SubnetId=subnet)
            if vpc:
                ec2.delete_vpc(VpcId=vpc)
        except Exception:
            pass
    parts = []
    for m in ("ondemand", "spot"):
        if m in found:
            parts.append(f"*** {m.upper()} CAPACITY in {found[m]} ***")
        elif quota.get(m):
            parts.append(f"{m}=QUOTA0")
        else:
            parts.append(f"{m}=none")
    return f"{region:15} -> {' | '.join(parts)}   (offered in {len(azs)} AZ)"


def main():
    print(f"probing {ITYPE} on-demand + spot across {len(REGIONS)} regions "
          f"(launch+immediate-terminate; unique tag)...\n")
    hits = []
    for region in REGIONS:
        line = probe_region(region)
        print(line, flush=True)
        if "CAPACITY in" in line:
            hits.append(line.split("->")[0].strip())
    print("\n== regions with ANY p4d capacity right now:", hits or "NONE", "==")
    print("(ONDEMAND -> aws_sweep_a100_shared.py --region <R>;  SPOT -> add spot market, ~70% cheaper)")


if __name__ == "__main__":
    main()
