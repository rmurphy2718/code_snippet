import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from multiprocessing.pool import Pool
from time import sleep

import redis


class M:
    LOC = 'mac-mk'
    COPA = "copa.cs.purdue.edu"
    LEBLON = "leblon.cs.purdue.edu"
    IPANEMA = "ipanema.cs.purdue.edu"
    URCA = "urca.cs.purdue.edu"
    M1 = "ml01.cs.purdue.edu"
    M2 = "ml02.cs.purdue.edu"
    M3 = "ml03.cs.purdue.edu"
    M4 = "ml04.cs.purdue.edu"
    M5 = "ml05.cs.purdue.edu"
    M6 = "ml06.cs.purdue.edu"
    M7 = "ml07.cs.purdue.edu"
    M8 = "ml08.cs.purdue.edu"
    GILBRETH = "gilbreth"


SCRATCH_FOLDER = {
    M.COPA: "scratch-data",
    M.LEBLON: "scratch-data",
    M.IPANEMA: "scratch-data",
    M.URCA: "scratch-data",
    M.GILBRETH: "scratch/gilbreth"
}
GPU_SPEC = {
    M.COPA: [0, 1, 2, 3],
    M.IPANEMA: [0, 1, 2, 3],
    M.URCA: [0, 1, 2, 3, 4, 5, 6, 7],
    M.LEBLON: [0],
    M.M1: [0, 1, 2, 3],
    M.M2: [0, 1, 2, 3],
    M.M3: [0, 1, 2, 3],
    M.M4: [0, 1, 2, 3],
    M.M5: [0, 1, 2, 3],
    M.M6: [0, 1, 2, 3],
    M.M7: [0, 1, 2, 3],
    M.M8: [0, 1, 2, 3],
    M.LOC: [0],
    M.GILBRETH: [0, 1],
}
RUNNING = "RUNNING"
MAX_WALL_CLOCK_TIME = 76300


def command_runner(c):
    subprocess.run(c, shell=True)


class Worker:

    @classmethod
    def check_create_dir(cls, d):
        if not os.path.exists(d):
            os.makedirs(d)
            print(f"Created d={d}")
        else:
            print(f"Exists d={d}")

    @classmethod
    def queue_baselines(cls, commands, rds, queue):
        for c in commands:
            print(c)
            rds.rpush(queue, json.dumps(c))

    @classmethod
    def clear(cls, rds, queue):
        return rds.delete(queue)

    @classmethod
    def len(cls, rds, queue):
        return rds.llen(queue)

    @classmethod
    def instantiate_workers(cls, gpu_spec=[0], proc_per_gpu=1, start=0, cluster=False, single=False):
        script = sys.argv[0].split("/")[-1]
        options = ""
        if cluster:
            options += " --cluster"
        if single:
            options += " --single"

        if cluster:
            commands = [f"{sys.executable} -u {script} worker --gpu_id {gpu_id} --ix {ix} {options}" for gpu_id in
                        gpu_spec for ix
                        in range(start, start + proc_per_gpu)]
            print(f"Running {len(commands)} commands in parallel")
            Pool(len(commands)).map(command_runner, commands)
            print(f"Join Threads")
        else:
            for gpu_id in gpu_spec:
                for ix in range(start, start + proc_per_gpu):
                    subprocess.run(
                        f"nohup {sys.executable} -u {script} worker --gpu_id {gpu_id} --ix {ix} {options}> workerhup.{gpu_id}.{ix}.out &",
                        shell=True)

    @classmethod
    def replace_macros(cls, input, macro_dict):
        input = str(input)
        for k, v in macro_dict.items():
            input = input.replace(k, str(v))
        return input

    @classmethod
    def worker(cls, act_machine, machine, rds, gpu_id, ix, single, input_queue, output_queue, error_queue):
        # single should be renamed to cluster
        time_budget_start = time.time()
        while True:
            start_time = time.time()
            ret_code = -1
            command = file = "none"
            try:
                pop_output = rds.blpop(input_queue, timeout=10)
                if pop_output is None:
                    break
                json_ct = pop_output[1]
                start_time = time.time()
                try:
                    rds.set(f"{RUNNING}:{act_machine}:{gpu_id}:{ix}", json_ct)
                except:
                    traceback.print_exc()
                command_tuple = json.loads(json_ct)
                working_dir, command, file = command_tuple[:3]
                macro_dict = {
                    "##SCRATCH##": ("scratch" if machine not in SCRATCH_FOLDER else SCRATCH_FOLDER[machine]),
                    "##GPU##": gpu_id,
                    "##SCRIPT##": "run/run.sh",
                    "##IDX##": str(ix),

                }

                working_dir = cls.replace_macros(working_dir, macro_dict)
                command = cls.replace_macros(command, macro_dict)
                file = f"{working_dir}/{cls.replace_macros(file, macro_dict)}"

                print(f'Running:{command}, output:{file}')
                with open(file, 'a') as fp:
                    if single:
                        total_time = MAX_WALL_CLOCK_TIME - (time.time() - time_budget_start)
                        try:
                            cp = subprocess.run(command, stdout=fp, stderr=fp, shell=True, cwd=working_dir,
                                                timeout=total_time)
                        except subprocess.TimeoutExpired:
                            rds.lpush(input_queue, json_ct)
                            break
                    else:
                        cp = subprocess.run(command, stdout=fp, stderr=fp, shell=True, cwd=working_dir)
                    ret_code = cp.returncode
                    if int(ret_code) == 0:
                        try:
                            rds.rpush(output_queue, json_ct)
                        except:
                            traceback.print_exc()
                    else:
                        try:
                            rds.rpush(error_queue, json_ct)
                        except:
                            traceback.print_exc()
            except:
                traceback.print_exc()
            finally:
                stop_time = time.time()
                rds.delete(f"RUNNING:{machine}:{gpu_id}:{ix}")
                print(f'Finished:{command}, output:{file}, time {stop_time - start_time}, ret_code={ret_code}')

    @classmethod
    def init(cls):
        parser = argparse.ArgumentParser("Producer Subscriber Queues")
        parser.add_argument("task", default="len", help=" One of [queue|clear|len|instantiate_workers|worker|requeue]")
        parser.add_argument('--single', default=False, action='store_const', const=True)
        parser.add_argument('--show_processes', default=False, action='store_const', const=True)
        parser.add_argument('--cluster', default=False, action='store_const', const=True)
        parser.add_argument('--start', default=0, type=int)
        parser.add_argument('--procs', default=1, type=int)
        parser.add_argument('--gpu_id', default=0, type=int)
        parser.add_argument('--ix', default=0, type=int)
        parser.add_argument('--gpu_spec', default=None, type=str)
        args = parser.parse_args()

        act_machine = machine = os.uname()[1]
        if machine.startswith(M.GILBRETH):
            machine = M.GILBRETH
        gpu_spec = GPU_SPEC[machine]

        if args.gpu_spec is not None:
            gpu_spec = list(map(lambda x: int(str(x).strip()), args.gpu_spec.split(",")))

        rds = redis.Redis("leblon.cs.purdue.edu", 16379,
                          password='bc7a102fb51acd91f33fbac48ec801e392fabe6ece4cbaf982cf3e8c0604eecb')
        return args, act_machine, machine, gpu_spec, rds

    @classmethod
    def requeue(cls, rds, error_queue, input_queue):
        while True:
            pop_output = rds.lpop(error_queue)
            if pop_output is None:
                break
            print(pop_output)
            rds.rpush(input_queue, pop_output)

    @classmethod
    def main(cls, TASKS, QUEUE_NAME):
        args, act_machine, machine, gpu_spec, rds = cls.init()

        input_queue = f"IQ:{QUEUE_NAME}:"
        output_queue = f"OQ:{QUEUE_NAME}:"
        error_queue = f"EQ:{QUEUE_NAME}:"

        if args.task == "clear":
            for q in [input_queue, output_queue, error_queue]:
                print(f"Clearing {q} = {cls.clear(rds, queue=q)}")
            for k in rds.keys(f"{RUNNING}*"):
                print(f"Clearing {k} = {cls.clear(rds, queue=k)}")
        elif args.task == "len":
            if args.show_processes:
                for q in [input_queue, output_queue, error_queue]:
                    print(f"Length of {q} = {cls.len(rds, queue=q)}")
                for k in sorted(rds.keys(f"{RUNNING}*")):
                    print(f"{str(k)} {rds.get(k)}")
            else:
                while True:
                    print("\t".join(map(lambda q: f"LEN({q})={cls.len(rds, queue=q)}", [input_queue, output_queue, error_queue])), end="\r",flush=True)
                    sleep(10)
        elif args.task == "queue":
            cls.queue_baselines(TASKS, rds, queue=input_queue)
        elif args.task == "requeue":
            cls.requeue(rds, error_queue, input_queue)
        elif args.task == "instantiate_workers":
            cls.instantiate_workers(gpu_spec=gpu_spec, proc_per_gpu=args.procs, start=args.start, cluster=args.cluster,
                                    single=args.single)
        elif args.task == "worker":
            # create workers
            cls.worker(act_machine, machine, rds, gpu_id=args.gpu_id, ix=args.ix, single=args.single or args.cluster,
                       input_queue=input_queue,
                       output_queue=output_queue,
                       error_queue=error_queue)
