import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path


def get_jobstate(job_id):
    cmd = f"sacct -j {job_id} -o state -n"
    p = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    ret = p.stdout.read().decode("utf8").strip()
    return ret


def get_data_type_and_part_id(filepath):
    path = Path(filepath)
    obj = re.search(r"tokenize-(.*?)-part-(\d+).log", path.name)
    if obj is None:
        return None
    data_type, part_id = obj.groups()
    return data_type, part_id


def check_result(filepath):
    path = Path(filepath)
    ret = get_data_type_and_part_id(filepath)
    if ret is None:
        return None
    data_type, part_id = ret
    content = path.read_text(encoding="utf8")

    if (
        "srun: error: Unable to allocate resources: Reach max user active rpc limit"
        in content
        or "srun: error: Unable to allocate resources: Socket timed out on send/recv operation"
        in content
    ):
        print(f"Error: {data_type}/{part_id}")
        return "error"

    obj = re.search(r"srun: job (\d+) queued and waiting for resources", content)
    if obj is None:
        print(f"Unknown: {data_type}/{part_id}")
        return "unknown"

    job_id = obj.group(1)
    jobstate = get_jobstate(job_id)
    obj = re.search(r"Tokenization Progress:\s*100%\s*\|.*\|\s*(\d+)/(\d+)", content)
    if obj is not None:
        progress, total = obj.groups()
        if (
            progress == total
            and progress is not None
            and total is not None
            and jobstate != "COMPLETED"
        ):
            print(f"DEAD_COMPLETED: {data_type}/{part_id} - job: {job_id}")
            return "DEAD_COMPLETED"

    print(f"{jobstate}: {data_type}/{part_id}")
    return jobstate


if __name__ == "__main__":
    status = defaultdict(list)
    for filepath in Path("logs").glob("tokenize-*.log"):
        s = check_result(filepath)
        res = get_data_type_and_part_id(filepath)
        status[s].append(res)

    print(Counter({k: len(v) for k, v in status.items()}).most_common())

    def print_val(v, k):
        print(f"# {k} = {len(v[k])}")
        for path in v[k]:
            print(path)

    for key in ["CANCELLED+", "DEAD_COMPLETED", "error", None]:
        print_val(status, key)
