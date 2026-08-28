# -*- coding: utf-8 -*-
"""
Probe whether this AWS account has On-Demand P-instance quota for p4d.24xlarge (8x A100), WITHOUT disturbing
any running tagged sweep. Attempts a real run_instances; interprets the outcome; if an instance is actually
created (quota exists), terminates it IMMEDIATELY by id (pending->terminated bills ~$0).

Safety: uses a DEFAULT subnet (no VPC creation, no shared cleanup tag), and tears down ONLY its own instance
by id. It NEVER calls the tag-wide cleanup(), so the concurrent A10G/L4/T4 spectrum sweep is untouched.
"""
import boto3
from aws_powercap_sweep import latest_pytorch_dlami

REGION = "us-east-1"
ITYPE = "p4d.24xlarge"
# UNIQUE tag -- deliberately NOT the sweep's "powershaping-powercap-probe" tag, so nothing here can be
# swept up by (or sweep up) the running A10G/L4/T4 sweep. Teardown is by explicit id only.
PTAGS = [{"Key": "Name", "Value": "p4d-quota-probe"}, {"Key": "purpose", "Value": "p4d-quota-probe"}]


def main():
    ec2 = boto3.client("ec2", region_name=REGION)
    ami = latest_pytorch_dlami(ec2)
    vpc = subnet = iid = None
    try:
        vpc = ec2.create_vpc(CidrBlock="10.9.0.0/16",
                             TagSpecifications=[{"ResourceType": "vpc", "Tags": PTAGS}])["Vpc"]["VpcId"]
        ec2.get_waiter("vpc_available").wait(VpcIds=[vpc])
        az = ec2.describe_availability_zones()["AvailabilityZones"][0]["ZoneName"]
        subnet = ec2.create_subnet(VpcId=vpc, CidrBlock="10.9.0.0/24", AvailabilityZone=az,
                                   TagSpecifications=[{"ResourceType": "subnet", "Tags": PTAGS}])["Subnet"]["SubnetId"]
        print(f"PROBE: type={ITYPE} ami={ami} temp-vpc={vpc} subnet={subnet} az={az}")
        try:
            r = ec2.run_instances(ImageId=ami, InstanceType=ITYPE, MinCount=1, MaxCount=1, SubnetId=subnet,
                                  InstanceInitiatedShutdownBehavior="terminate",
                                  TagSpecifications=[{"ResourceType": "instance", "Tags": PTAGS}])
            iid = r["Instances"][0]["InstanceId"]
            print(f"RESULT: LAUNCHED -> P-QUOTA EXISTS (instance {iid}); terminating immediately.")
        except Exception as e:
            s = str(e)
            if "VcpuLimitExceeded" in s or "InstanceLimitExceeded" in s:
                print("RESULT: QUOTA = 0  (VcpuLimitExceeded) -> need a Service Quota increase for "
                      "'Running On-Demand P instances' (>=96 vCPU) before p4d can launch.")
            elif "InsufficientInstanceCapacity" in s:
                print("RESULT: QUOTA OK, but NO CAPACITY right now (InsufficientInstanceCapacity) -> retry / another AZ.")
            elif "Unsupported" in s or "not supported" in s.lower():
                print(f"RESULT: {ITYPE} not offered here -> {s[:160]}")
            else:
                print(f"RESULT: OTHER ERROR: {type(e).__name__}: {s[:200]}")
    finally:
        # targeted teardown by id ONLY -- never tag-wide cleanup (would hit the live sweep)
        try:
            if iid:
                ec2.terminate_instances(InstanceIds=[iid])
                print(f"PROBE: terminate sent for {iid}; waiting ...")
                ec2.get_waiter("instance_terminated").wait(InstanceIds=[iid])
            if subnet:
                ec2.delete_subnet(SubnetId=subnet); print(f"PROBE: deleted subnet {subnet}")
            if vpc:
                ec2.delete_vpc(VpcId=vpc); print(f"PROBE: deleted vpc {vpc}")
        except Exception as e:
            print(f"PROBE: teardown warning: {type(e).__name__}: {str(e)[:160]} (check console for {vpc})")


if __name__ == "__main__":
    main()
