from fastapi import FastAPI, Depends, Request
from fastapi.exceptions import HTTPException
import pydantic
import asyncio
import bittensor as bt
import uvicorn
from concurrent.futures import ThreadPoolExecutor
import random
import httpx
from threading import Thread
import time
from ..constants import constants
from ..protocol import TextCompressProtocol
from ..validator_utils import MinerManager


class OrganicPayload(pydantic.BaseModel):
    context: str
    tier: str
    target_model: str
    miner_uid: int = -1
    top_incentive: float = 0.9


class OrganicResponse(pydantic.BaseModel):
    compressed_tokens_b64: str


class RegisterPayload(pydantic.BaseModel):
    port: int


class OrganicGate:
    def __init__(
        self,
        miner_manager: MinerManager,
        wallet,
        config: bt.config,
        metagraph,
    ):
        self.metagraph: bt.metagraph.__class__ = metagraph
        self.miner_manager = miner_manager
        self.wallet = wallet
        self.config = config
        self.dendrite = bt.dendrite(wallet=wallet)
        self.app = FastAPI()
        self.app.add_api_route(
            "/forward",
            self.forward,
            methods=["POST"],
            dependencies=[Depends(self.get_self)],
        )
        self.app.add_api_route(
            "/health",
            self.health_check,
            methods=["GET"],
            dependencies=[Depends(self.get_self)],
        )
        self.loop = asyncio.get_event_loop()
        self.client_axon: bt.AxonInfo = None
        self.message = "".join(random.choices("0123456789abcdef", k=16))
        self.start_server()
        self.register_to_client()
        self.sync_thread = Thread(
            target=self._run_function_periodically,
            args=(self.register_to_client, 60),
        )
        self.sync_thread.start()

    def _run_function_periodically(self, function, interval):
        while True:
            bt.logging.info(
                f"Running function {function.__name__} every {interval} seconds."
            )
            try:
                function()
            except Exception as e:
                bt.logging.error(f"Error running function {function.__name__}: {e}")
            time.sleep(interval)

    def register_to_client(self):
        payload = RegisterPayload(port=self.config.validator.gate_port)
        self.call(payload, timeout=12)

    async def _authenticate(self, request: Request):
        message = request.headers["message"]
        if message != self.message:
            raise Exception("Authentication failed.")

    async def forward(self, request: Request):
        try:
            await self._authenticate(request)
            bt.logging.info("Forwarding organic request.")
            request: OrganicPayload = OrganicPayload(**await request.json())
            synapse = TextCompressProtocol(
                context=request.context,
                target_model=request.target_model,
            )
            bt.logging.info(f"Context: {request.context[:100]}...")
            bt.logging.info(f"Tier: {request.tier}")
            bt.logging.info(f"Target model: {request.target_model}")

            targeted_uid = None
            if request.miner_uid != -1:
                counter = self.miner_manager.serving_counter[request.tier][
                    request.miner_uid
                ]
                if counter.increment():
                    targeted_uid = request.miner_uid
            else:
                for uid, counter in self.miner_manager.serving_counter[
                    request.tier
                ].items():
                    if counter.increment():
                        targeted_uid = uid
                        break
            if targeted_uid is None:
                raise HTTPException(
                    status_code=503,
                    detail="No miners available.",
                )
            target_axon = self.metagraph.axons[targeted_uid]

            response: TextCompressProtocol = await self.dendrite.forward(
                axons=target_axon,
                synapse=synapse,
                timeout=constants.TIER_CONFIG[request.tier].timeout,
                deserialize=False,
            )
            response.base64_to_ndarray()
            bt.logging.info(f"Compressed shape: {response.compressed_tokens.shape}")
        except Exception as e:
            bt.logging.error(f"Error: {e}")
            raise HTTPException(status_code=503, detail="Validator error.")

        return OrganicResponse(compressed_tokens_b64=response.compressed_tokens_b64)

    def start_server(self):
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.executor.submit(
            uvicorn.run,
            self.app,
            host="0.0.0.0",
            port=self.config.validator.gate_port,
        )

    async def get_self(self):
        return self

    async def health_check(self):
        return {"status": "healthy"}

    def call(
        self,
        payload: RegisterPayload,
        timeout: float = 12.0,
    ) -> bt.Synapse:
        """
        Customized call method to send Synapse-like requests to the Organic Client Server.

        Args:
            dendrite (bt.Dendrite): The Dendrite object to send the request.
            url (str): The URL of the Organic Client Server.
            payload (pydantic.BaseModel): The payload to send in the request.
            timeout (float, optional): Maximum duration to wait for a response from the Axon in seconds. Defaults to ``12.0``.

        Returns:

        """

        url = f"{self.config.validator.organic_client_url}/register"
        signature = f"0x{self.dendrite.keypair.sign(self.message).hex()}"

        headers = {
            "Content-Type": "application/json",
            "message": self.message,
            "ss58_address": self.wallet.hotkey.ss58_address,
            "signature": signature,
        }

        with httpx.Client() as client:
            response = client.post(
                url,
                json=payload.model_dump(),
                headers=headers,
                timeout=timeout,
            )

        if response.status_code != 200:
            bt.logging.error(
                f"Failed to register to the Organic Client Server. Response: {response.text}"
            )
            return
        else:
            bt.logging.success("Registered to the Organic Client Server.")
            return response.json()
