"""Entry point to multi-node (distributed) training for a user given experiment
name."""

import os
import random
import string
import subprocess
import sys
import time
import re
from pathlib import Path
from typing import Optional

# Add to PYTHONPATH the path of the parent directory of the current file's directory
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(Path(__file__)))))

from allenact.main import get_argument_parser as get_main_arg_parser
from allenact.utils.system import init_logging, get_logger, find_free_port
from home_service_constants import ABS_PATH_OF_HOME_SERVICE_TOP_LEVEL_DIR
from dscripts.sshutils import read_openssh_config


def get_argument_parser():
    """Creates the argument parser."""

    parser = get_main_arg_parser()
    parser.description = f"distributed {parser.description}"

    parser.add_argument(
        "--runs_on",
        required=True,
        type=str,
        help="Comma-separated IP addresses of machines",
    )

    parser.add_argument(
        "--master",
        type=str,
        default="mil",
    )

    parser.add_argument(
        "--ssh_cmd",
        required=False,
        type=str,
        default="ssh -f {addr}",
        help="SSH command. Useful to utilize a pre-shared key with 'ssh -i mykey.pem -f ubuntu@{addr}'. "
        "The option `-f` should be used for non-interactive session",
    )

    parser.add_argument(
        "--conda_env",
        required=True,
        type=str,
        help="Name of the conda environment. It must be the same across all machines",
    )

    parser.add_argument(
        "--home_service_path",
        required=False,
        type=str,
        default="robot_home_service",
        help="Path to home_service top directory. It must be the same across all machines",
    )

    # Required distributed_ip_and_port
    idx = [a.dest for a in parser._actions].index("distributed_ip_and_port")
    parser._actions[idx].required = True

    return parser


def get_args():
    """Creates the argument parser and parses any input arguments."""

    parser = get_argument_parser()
    args = parser.parse_args()

    return args


def get_raw_args():
    raw_args = sys.argv[1:]
    filtered_args = []
    remove: Optional[str] = None
    enclose_in_quotes: Optional[str] = None
    for arg in raw_args:
        if remove is not None:
            remove = None
        elif enclose_in_quotes is not None:
            # Within backslash expansion: close former single, open double, create single, close double, reopen single
            inner_quote = r"\'\"\'\"\'"
            # Convert double quotes into backslash double for later expansion
            filtered_args.append(
                inner_quote + arg.replace('"', r"\"").replace("'", r"\"") + inner_quote
            )
            enclose_in_quotes = None
        elif arg in [
            "--runs_on",
            "--ssh_cmd",
            "--conda_env",
            "--home_service_path",
            "--extra_tag",
            "--machine_id",
            "--master",
        ]:
            remove = arg
        elif arg == "--config_kwargs":
            enclose_in_quotes = arg
            filtered_args.append(arg)
        else:
            filtered_args.append(arg)
    return filtered_args


def wrap_single(text):
    return f"'{text}'"


def wrap_single_nested(text):
    # Close former single, start backslash expansion (via $), create new single quote for expansion:
    quote_enter = r"'$'\'"
    # New closing single quote for expansion, close backslash expansion, reopen former single:
    quote_leave = r"\'''"
    return f"{quote_enter}{text}{quote_leave}"


def wrap_double(text):
    return f'"{text}"'


def id_generator(size=4, chars=string.ascii_uppercase + string.digits):
    return "".join(random.choice(chars) for _ in range(size))


# Assume we can ssh into each of the `runs_on` machines through port 22
if __name__ == "__main__":
    # Tool must be called from Rearrange2022 project's root directory
    cwd = os.path.abspath(os.getcwd())
    assert cwd == ABS_PATH_OF_HOME_SERVICE_TOP_LEVEL_DIR, (
        f"`dmain.py` called from {cwd}."
        f"\nIt should be called from Rearrange2022's top level directory {ABS_PATH_OF_HOME_SERVICE_TOP_LEVEL_DIR}."
    )

    args = get_args()

    init_logging(args.log_level)

    raw_args = get_raw_args()

    if args.seed is None:
        seed = random.randint(0, 2 ** 31 - 1)
        raw_args.extend(["-s", f"{seed}"])
        get_logger().info(f"Using random seed {seed} in all workers (none was given)")

    all_addresses = args.runs_on.split(",")
    get_logger().info(f"Running on IP addresses {all_addresses}")

    assert args.distributed_ip_and_port.split(":")[0] in all_addresses, (
        f"Missing listener IP address {args.distributed_ip_and_port.split(':')[0]}"
        f" in list of worker addresses {all_addresses}"
    )

    time_str = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(time.time()))

    global_job_id = id_generator()
    killfilename = os.path.join(
        os.path.expanduser("~"), ".allenact", f"{time_str}_{global_job_id}.killfile"
    )
    os.makedirs(os.path.dirname(killfilename), exist_ok=True)

    code_src = "."

    with open(killfilename, "w") as killfile:
        for it, addr in enumerate(all_addresses):
            ip_regex = re.compile(r'^\d{1,3}\.\d{1,3}\.\d{1,3}.\d{1,3}')
            if re.match(ip_regex, args.distributed_ip_and_port.split(":")[0]):
                distributed_ip = args.distributed_ip_and_port.split(":")[0]
            else:
                if addr != args.master:
                    distributed_ip, _, _, _ = read_openssh_config(args.distributed_ip_and_port.split(":")[0])
                else:
                    distributed_ip = "127.0.0.1"

            if (
                len(args.distributed_ip_and_port.split(":")) == 1
                or len(args.distributed_ip_and_port.split(":")[1]) == 0
                or int(args.distributed_ip_and_port.split(":")[1]) == 0
            ):
                distributed_port = find_free_port()
            else:
                if addr != args.master:
                    distributed_port = int(args.distributed_ip_and_port.split(":")[1])
                else:
                    distributed_port = 5000
            
            idx = raw_args.index('--distributed_ip_and_port')
            raw_args[idx+1] = f"{distributed_ip}:{distributed_port}"

            if addr != args.master:
                code_tget = f"{addr}:{args.home_service_path}/"
                get_logger().info(f"rsync {code_src} to {code_tget}")
                os.system(f"rsync -rz {code_src} {code_tget} --exclude '__pycache__' --exclude '{args.output_dir}'")

            job_id = id_generator()

            command = " ".join(
                ["allenact"]
                + raw_args
                + [
                    "--extra_tag",
                    f"{args.extra_tag}{'__' if len(args.extra_tag) > 0 else ''}machine{it}",
                ]
                + ["--machine_id", f"{it}"]
            )

            logfile = (
                f"{args.output_dir}/log_{time_str}_{global_job_id}_{job_id}_machine{it}"
            )

            env_and_command = wrap_single_nested(
                f"for NCCL_SOCKET_IFNAME in $(route | grep default) ; do : ; done && export NCCL_SOCKET_IFNAME"
                f" && export NCCL_P2P_DISABLE=1"
                f" && cd {args.home_service_path}"
                f" && export PYTHONPATH=$PYTHONPATH:$PWD"
                f" && mkdir -p {args.output_dir}"
                f" && conda_base"
                f" && conda activate {args.conda_env} &>> {logfile}"
                f" && echo pwd=$(pwd) &>> {logfile}"
                f" && echo output_dir={args.output_dir} &>> {logfile}"
                f" && echo python_version=$(python --version) &>> {logfile}"
                f" && echo python_path=$(which python) &>> {logfile}"
                f" && set | grep NCCL_SOCKET_IFNAME &>> {logfile}"
                f" && set | grep NCCL_P2P_DISABLE &>> {logfile}"
                f" && set | grep PYTHONPATH &>> {logfile}"
                f" && echo allenact_path=$(which allenact) &>> {logfile}"
                f" && echo &>> {logfile}"
                f" && {command} &>> {logfile}"
            )

            tmux_name = f"allenact_{time_str}_{global_job_id}_{job_id}_machine{it}"
            tmux_new_command = wrap_single(
                f"tmux new-session -s {tmux_name} -d && tmux send-keys -t {tmux_name} {env_and_command} C-m"
            )
            ssh_command = f"{args.ssh_cmd.format(addr=addr)} {tmux_new_command}"
            get_logger().debug(f"SSH command {ssh_command}")
            subprocess.run(ssh_command, shell=True, executable="/bin/bash")
            get_logger().info(f"{addr} {tmux_name}")

            killfile.write(f"{addr} {tmux_name}\n")

    get_logger().info("")
    get_logger().info(f"Running tmux ids saved to {killfilename}")
    get_logger().info("")

    get_logger().info("DONE")
