import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200)
parser.add_argument("--k1", type=int, default=9, help="intra_modal neighbor")
parser.add_argument(
    "--k2", type=int, default=10, help="help='inra_modal neighbor'"
)  # 40
parser.add_argument("--k3", type=int, default=10, help="inter_modal neighbor")
parser.add_argument("--batch_size", type=int, default=128)  # 128
parser.add_argument(
    "--lr_g", type=float, default=5e-4, help="adam: learning rate for G"  # 1e-4
)
parser.add_argument(
    "--lr_m", type=float, default=1e-4, help="adam: learning rate for model"  # 1e-4
)
parser.add_argument(
    "--lr_ae", type=float, default=1e-5, help="adam: learning rate for AE"  # 1e-4
)
parser.add_argument(
    "--b1",
    type=float,
    default=0.5,
    help="adam: decay of first order momentum of gradient",
)
parser.add_argument(
    "--b2",
    type=float,
    default=0.999,
    help="adam: decay of first order momentum of gradient",
)
parser.add_argument("--weight_decay", type=float, default=0)  # 1e-4
parser.add_argument(
    "--lamda1", type=float, default=1.0, help="reg for cycle consistency"
)
parser.add_argument(
    "--lamda2", type=float, default=0.01, help="reg for adversarial loss"  # 1.0
)
parser.add_argument("--clip_value", type=float, default=0.05, help="gradient clipping")

parser.add_argument(
    "--n_cpu", type=int, default=8, help="# of cpu threads during batch generation"
)
parser.add_argument("--seed", type=int, default=2023)
parser.add_argument(
    "--pretrain",
    type=str,
    default="load_all",
    choices=["img", "txt", "load_ae", "load_all", "None"],
)
parser.add_argument(
    "--dataset", type=str, default="wikipedia", choices=["wikipedia", "nuswide"]
)
parser.add_argument("--data_dir", type=str, default="data/wikipedia/")
parser.add_argument(
    "--cpt_dir", type=str, default="cpt/", help="dir for saved checkpoint"
)
parser.add_argument(
    "--img_cptpath",
    type=str,
    default="cpt/wikipedia_img_pretrain_checkpt.pkl",
    help="path to load img AE checkpoint",
)
parser.add_argument(
    "--txt_cptpath",
    type=str,
    default="cpt/wikipedia_txt_pretrain_checkpt.pkl",
    help="path to load txt AE checkpoint",
)
parser.add_argument(
    "--dm2c_cptpath",
    type=str,
    default="cpt/wikipedia_checkpt.pkl",
    help="path to load dm2c checkpoint",
)
args = parser.parse_args()
