import argparse

parser = argparse.ArgumentParser()

# "template"
parser.add_argument("--foo", type=int, default=7)
parser.add_argument("--bar", type=str, default="bar")
parser.add_argument("--bool",
                    type=str,
                    default="false",
                    choices=["true", "false"])

# "train params"
parser.add_argument("--epos", type=int, default=300)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument('--lr_decay', type=float, default=0.995)
parser.add_argument("--wd", type=float, default=1e-4)
parser.add_argument('--opt',
                    type=str,
                    choices=["adam", "adamw", "sgd"],
                    default="adamw")
parser.add_argument("--bs", type=int, default=128)
parser.add_argument("--bs_dev", type=int, default=512)
parser.add_argument("--gpus", type=str, default="0")
parser.add_argument("--nogpu",
                    type=str,
                    default="false",
                    choices=["true", "false"])
parser.add_argument("--tpu",
                    type=str,
                    default="false",
                    choices=["true", "false"])
parser.add_argument("--lr_find",
                    type=str,
                    default="false",
                    choices=["true", "false"])
parser.add_argument("--model",
                    type=str,
                    choices=["Model0", "Model1"],
                    default="Model0")
parser.add_argument("--check_val", type=int, default=5)

# "program params"
parser.add_argument("--data_pt", type=str, default="data/data.h5")
parser.add_argument("--dbg",
                    type=str,
                    default="false",
                    choices=["true", "false"])
parser.add_argument("--test",
                    type=str,
                    default="false",
                    choices=["true", "false"])
parser.add_argument("--log_file", type=str, default="runs/logs/train")
parser.add_argument("--log_level",
                    type=str,
                    choices=["info", "debug", "warning"],
                    default="debug")
parser.add_argument("--pb_rate", type=int, default=20)
parser.add_argument("--nw0", type=int, default=4)
parser.add_argument("--nw1", type=int, default=8)

parser.add_argument("--tune_name", type=str, default="tune_0")
parser.add_argument("--tune_schd",
                    type=str,
                    default="asha",
                    choices=["asha", "pbt"])
parser.add_argument("--tune_num_samples", type=int, default=8)
parser.add_argument("--tune_num_cpus", type=int, default=16)
parser.add_argument("--tune_num_gpus", type=int, default=2)
parser.add_argument("--tune_gups", type=str, default="0,1")
parser.add_argument("--tune_per_cpu", type=float, default=4)
parser.add_argument("--tune_per_gpu", type=float, default=0.5)

# "model params"


def _bool(x):
    assert x in ("true", "false"), "the x is {}".format(x)
    return True if x == "true" else False


def _convert(config):
    for k, v in config.__dict__.items():
        if v in ("true", "false"):
            config.__dict__[k] = _bool(v)
        if v == "None" or v == "none":
            config.__dict__[k] = None


config, _ = parser.parse_known_args()
_convert(config)

if __name__ == "__main__":
    print(config)
