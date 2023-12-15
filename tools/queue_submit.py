import re
import time
import subprocess
from pathlib import Path

from loguru import logger

from smoe.utils.notification import send_to_wechat

from check_killed import get_jobstate


logger.add("logs/queue_submit.log")


def run_command(command):
    try:
        logger.info(f"Running cmd: {command}")
        subprocess.run(
            command,
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        logger.info(f"An error occurred: {e}")


def get_jobid(filepath):
    content = Path(filepath).read_text(encoding="utf8")
    obj = re.search(r"srun: job (\d+) queued and waiting for resources", content)
    if obj is None:
        return None
    job_id = obj.group(1)
    return job_id


if __name__ == "__main__":
    task_list = [
        # CANCELLED+
        ('en_stack', '000028'),
        ('en_wikipedia', '000004'),
        ('en_wikipedia', '000006'),
        ('en_wikipedia', '000009'),
        ('github', '000024'),
        ('github', '000029'),
        ('en_wikipedia', '000010'),
        ('en_wikipedia', '000012'),
        ('en_wikipedia', '000024'),
        ('en_wikipedia', '000026'),
        ('en_wikipedia', '000029'),
        ('github', '000004'),
        ('en_wikipedia', '000000'),
        ('en_wikipedia', '000002'),
        ('github', '000014'),
        ('github', '000016'),
        ('github', '000019'),
        ('github', '000020'),
        ('github', '000022'),
        # error = 10,
        ('github', '000011'),
        ('github', '000013'),
        ('github', '000027'),
        ('github', '000007'),
        ('github', '000008'),
        ('github', '000010'),
        ('github', '000012'),
        ('github', '000026'),
        ('github', '000006'),
        ('github', '000009'),
        # un-processed,
        ("en_cc", "000000"),
        ("en_cc", "000001"),
        ("en_cc", "000002"),
        ("en_cc", "000003"),
        ("en_cc", "000004"),
        ("en_cc", "000005"),
        ("en_cc", "000006"),
        ("en_cc", "000007"),
        ("en_cc", "000008"),
        ("en_cc", "000009"),
        ("en_cc", "000010"),
        ("en_cc", "000011"),
        ("en_cc", "000012"),
        ("en_cc", "000013"),
        ("en_cc", "000014"),
        ("en_cc", "000015"),
        ("en_cc", "000016"),
        ("en_cc", "000017"),
        ("en_cc", "000018"),
        ("en_cc", "000019"),
        ("en_cc", "000020"),
        ("en_cc", "000021"),
        ("en_cc", "000022"),
        ("en_cc", "000023"),
        ("en_cc", "000024"),
        ("en_cc", "000025"),
        ("en_cc", "000026"),
        ("en_cc", "000027"),
        ("en_cc", "000028"),
        ("en_cc", "000029"),
    ]
    data_dir = Path(
        "/mnt/petrelfs/share_data/zhutong/data/slimpajama_fluency_llama_middle_parts"
    )
    out_dir = Path(
        "/mnt/petrelfs/share_data/zhutong/data/slimpajama_fluency_mistral_middle_parts"
    )

    def submit_job(data_type, part_id):
        run_command(
            f"nohup srun -p MoE_T -N1 -n1 -c 32 "
            f"python -m smoe.utils.tokenize "
            f"-f jsonl -c input_ids "
            "-s /mnt/petrelfs/share_data/zhutong/models/llama2_7B "
            "-t /mnt/petrelfs/share_data/zhutong/models/Mistral-7B-v0.1 "
            f"-i {str(data_dir / data_type / f'part-{part_id}')} "
            f"-o {str(out_dir / data_type / f'part-{part_id}')} "
            f"1>logs/tokenize-{data_type}-part-{part_id}.log 2>&1 &"
        )

    wait_seconds = 5
    check_times = 10
    while len(task_list) > 0:
        data_type, part_id = task_list.pop(0)
        filepath = data_dir / data_type / f"part-{part_id}"
        logger.info(f"Processing: {filepath}")
        submit_job(data_type, part_id)
        check_times = 10
        time.sleep(5)
        jobstate = "PENDING"
        while jobstate != "RUNNING" and jobstate != "COMPLETED":
            if "CANCELLED" in jobstate:
                send_to_wechat(f"Job {data_type}-{part_id} is cancelled, resubmit")
                submit_job(data_type, part_id)
            if check_times <= 0:
                wait_seconds = 600
            time.sleep(wait_seconds)
            job_id = get_jobid(f"logs/tokenize-{data_type}-part-{part_id}.log")
            jobstate = get_jobstate(job_id)
            logger.info(f"Check job: {job_id} - {jobstate}")
            check_times -= 1
