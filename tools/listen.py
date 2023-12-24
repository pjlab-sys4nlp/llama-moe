import subprocess
import time

from smoe.utils.notification import send_to_wechat


def check_sme_pending():
    # run sme | grep "normal PD" | wc -l, if the returned value is 0, then send a notification
    cmd = "squeue --me | grep 'normal PD' | wc -l"
    p = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    for line in p.stdout.readlines():
        line = line.decode("utf-8")
        if int(line) == 0:
            send_to_wechat("pending jobs all clear!!!")
            return True
    return False


def check_sme_running():
    # run sme | grep "normal  R" | wc -l, if the returned value is 0, then send a notification
    cmd = "squeue --me | grep 'normal  R' | wc -l"
    p = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    for line in p.stdout.readlines():
        line = line.decode("utf-8")
        if int(line) == 0:
            send_to_wechat("running jobs all clear!!!")
            return True
    return False


def listen():
    # check pending jobs every 10 seconds, if all pending jobs are done, send a notification
    no_pending = False
    no_running = False
    while True:
        if not no_pending:
            no_pending = check_sme_pending()
            time.sleep(10)
        if not no_running:
            no_running = check_sme_running()
            time.sleep(10)


if __name__ == "__main__":
    listen()
