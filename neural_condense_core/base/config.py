import bittensor as bt
from argparse import ArgumentParser
from ..constants import constants


def add_common_config(parser: ArgumentParser):
    parser.add_argument("--netuid", type=int, default=1, help="The chain subnet uid.")
    parser.add_argument(
        "--whitelist_uids",
        type=str,
        default=None,
        help="The uids to whitelist. For testing purposes.",
    )
    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)
    bt.wallet.add_args(parser)
    bt.axon.add_args(parser)
    return parser


def add_validator_config(parser: ArgumentParser):
    parser.add_argument(
        "--validator.gate_port",
        type=int,
        default=None,
        help="The port of the validator gate server.",
    )

    parser.add_argument(
        "--validator.score_backend.host",
        type=str,
        default="localhost",
        help="The host of the score backend server.",
    )
    parser.add_argument(
        "--validator.score_backend.port",
        type=int,
        default=8089,
        help="The port of the score backend server.",
    )

    parser.add_argument(
        "--validator.organic_client_url",
        type=str,
        default=constants.ORGANIC_CLIENT_URL,
        help="The URL of the organic client.",
    )

    parser.add_argument(
        "--validator.report_url",
        type=str,
        default=constants.REPORT_URL,
        help="The URL of the report server.",
    )

    parser.add_argument(
        "--validator.use_wandb",
        action="store_true",
        help="Whether to use wandb for logging.",
    )

    return parser


def add_miner_config(parser: ArgumentParser):
    tier_names = list(constants.TIER_CONFIG.keys())
    parser.add_argument(
        "--miner.backend_host",
        type=str,
        default="localhost",
        help="The host of the backend server.",
    )
    parser.add_argument(
        "--miner.backend_port",
        type=int,
        default=8088,
        help="The port of the backend server.",
    )
    parser.add_argument(
        "--miner.tier",
        choices=tier_names,
    )
    return parser
