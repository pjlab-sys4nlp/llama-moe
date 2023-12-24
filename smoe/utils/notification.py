import datetime
import functools
import json
import logging
import os
import socket
import traceback

import dotenv
import requests

dotenv.load_dotenv()

logger_name = "Notifier - RANK: %s" % os.environ.get("RANK", "None")
logger = logging.getLogger(logger_name)

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_slurm_job_name():
    job_name = os.environ.get("SLURM_JOB_NAME", "SLURM_JOB_NAME")
    job_id = os.environ.get("SLURM_JOB_ID", "SLURM_JOB_ID")
    return f"{job_name}-{job_id}"


def send_to_wechat(
    msg: str,
    webhook_url: str = None,
    user_mentions: list[str] = None,
    user_mentions_mobile: list[str] = None,
):
    if not webhook_url:
        webhook_url = os.environ.get("WECHAT_ROBOT_WEBHOOK")
    if not user_mentions:
        env_user_mentions = os.environ.get("WECHAT_ROBOT_MENTIONS", "")
        user_mentions = env_user_mentions.split(",")
    if not user_mentions_mobile:
        env_user_mentions_mobile = os.environ.get("WECHAT_ROBOT_MENTIONS_MOBILE", "")
        user_mentions_mobile = env_user_mentions_mobile.split(",")

    msg_template = {
        "msgtype": "text",
        "text": {
            "content": msg,
            "mentioned_list": user_mentions,
            "mentioned_mobile_list": user_mentions_mobile,
        },
    }
    requests.post(webhook_url, json=msg_template)


def wechat_sender(
    webhook_url: str = None,
    user_mentions: list[str] = [],
    user_mentions_mobile: list[str] = [],
    msg_prefix: str = "",
):
    """
    WeChat Work sender wrapper: execute func, send a WeChat Work notification with the end status
    (sucessfully finished or crashed) at the end. Also send a WeChat Work notification before
    executing func. To obtain the webhook, add a Group Robot in your WeChat Work Group. Visit
    https://work.weixin.qq.com/api/doc/90000/90136/91770 for more details.
    Credit to: https://github.com/huggingface/knockknock/blob/master/knockknock/wechat_sender.py
    The original code snippet is distributed under MIT license.

    `webhook_url`: str
        The webhook URL to access your WeChat Work chatroom.
        Visit https://work.weixin.qq.com/api/doc/90000/90136/91770 for more details.
    `user_mentions`: List[str] (default=[])
        Optional userids to notify (use '@all' for all group members).
        Visit https://work.weixin.qq.com/api/doc/90000/90136/91770 for more details.
    `user_mentions_mobile`: List[str] (default=[])
        Optional user's phone numbers to notify (use '@all' for all group members).
        Visit https://work.weixin.qq.com/api/doc/90000/90136/91770 for more details.

    """

    if not webhook_url:
        webhook_url = os.environ.get("WECHAT_ROBOT_WEBHOOK")
    if not user_mentions:
        env_user_mentions = os.environ.get("WECHAT_ROBOT_MENTIONS", "")
        user_mentions = env_user_mentions.split(",")
    if not user_mentions_mobile:
        env_user_mentions_mobile = os.environ.get("WECHAT_ROBOT_MENTIONS_MOBILE", "")
        user_mentions_mobile = env_user_mentions_mobile.split(",")

    msg_template = {
        "msgtype": "text",
        "text": {
            "content": "",
            "mentioned_list": user_mentions,
            "mentioned_mobile_list": user_mentions_mobile,
        },
    }

    def decorator_sender(func):
        @functools.wraps(func)
        def wrapper_sender(*args, **kwargs):
            start_time = datetime.datetime.now()
            host_name = socket.gethostname()
            func_name = func.__name__

            # Handling distributed training edge case.
            # In PyTorch, the launch of `torch.distributed.launch` sets up a RANK environment variable for each process.
            # This can be used to detect the master process.
            # See https://github.com/pytorch/pytorch/blob/master/torch/distributed/launch.py#L211
            # Except for errors, only the master process will send notifications.
            if "RANK" in os.environ:
                master_process = int(os.environ["RANK"]) == 0
                host_name += " - RANK: %s" % os.environ["RANK"]
            else:
                master_process = True

            if master_process:
                contents = [
                    "Your training has started 🎬",
                    "Machine name: %s" % host_name,
                    "Main call: %s" % func_name,
                    f"Job {get_slurm_job_name()}",
                    "Starting date: %s" % start_time.strftime(DATE_FORMAT),
                ]

                msg_template["text"]["content"] = f"{msg_prefix}\n" + "\n".join(
                    contents
                )
                logger.info(f"{json.dumps(msg_template, ensure_ascii=False)}")
                if webhook_url:
                    requests.post(webhook_url, json=msg_template)

                try:
                    value = func(*args, **kwargs)

                    if master_process:
                        end_time = datetime.datetime.now()
                        elapsed_time = end_time - start_time
                        contents = [
                            "Your training is complete 🎉",
                            "Machine name: %s" % host_name,
                            "Main call: %s" % func_name,
                            f"Job {get_slurm_job_name()}",
                            "Starting date: %s" % start_time.strftime(DATE_FORMAT),
                            "End date: %s" % end_time.strftime(DATE_FORMAT),
                            "Training duration: %s" % str(elapsed_time),
                        ]

                        try:
                            str_value = str(value)
                            contents.append("Main call returned value: %s" % str_value)
                        except Exception:
                            contents.append(
                                "Main call returned value: %s"
                                % "ERROR - Couldn't str the returned value."
                            )

                        msg_template["text"]["content"] = f"{msg_prefix}\n" + "\n".join(
                            contents
                        )
                        logger.info(f"{json.dumps(msg_template, ensure_ascii=False)}")
                        if webhook_url:
                            requests.post(webhook_url, json=msg_template)

                    return value

                except Exception as ex:
                    end_time = datetime.datetime.now()
                    elapsed_time = end_time - start_time
                    contents = [
                        "Your training has crashed ☠️",
                        "Machine name: %s" % host_name,
                        "Main call: %s" % func_name,
                        f"Job {get_slurm_job_name()}",
                        "Starting date: %s" % start_time.strftime(DATE_FORMAT),
                        "Crash date: %s" % end_time.strftime(DATE_FORMAT),
                        "Crashed training duration: %s\n\n" % str(elapsed_time),
                        "Here's the error:",
                        "%s\n\n" % ex,
                        "Traceback:",
                        "%s" % traceback.format_exc(),
                    ]

                    msg_template["text"]["content"] = f"{msg_prefix}\n" + "\n".join(
                        contents
                    )
                    logger.info(f"{json.dumps(msg_template, ensure_ascii=False)}")
                    if webhook_url:
                        requests.post(webhook_url, json=msg_template)

                    raise ex

        return wrapper_sender

    return decorator_sender
