from fastapi import FastAPI, Depends, Request
from fastapi.exceptions import HTTPException
import pydantic
import asyncio
import bittensor as bt
import uvicorn
from concurrent.futures import ThreadPoolExecutor
import random
import httpx
import time
from ...constants import constants
from ...protocol import TextCompressProtocol
from ..managing import MinerManager
from ...logger import logger


class OrganicPayload(pydantic.BaseModel):
    context: str
    tier: str
    target_model: str
    miner_uid: int = -1
    top_incentive: float = 0.9


class OrganicResponse(pydantic.BaseModel):
    compressed_kv_url: str
    miner_uid: int = -1
    compressed_text: str


class RegisterPayload(pydantic.BaseModel):
    port: int


class OrganicGate:
    def __init__(
        self,
        miner_manager: MinerManager,
        config: bt.config,
    ):
        self.metagraph: bt.metagraph.__class__ = miner_manager.metagraph
        self.miner_manager = miner_manager
        self.wallet = miner_manager.wallet
        self.config = config
        self.dendrite = bt.dendrite(wallet=miner_manager.wallet)
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
        self.client_axon: bt.AxonInfo = None
        self.authentication_key = "".join(random.choices("0123456789abcdef", k=16))

    async def _run_function_periodically(self, function, interval):
        while True:
            logger.info(
                f"Running function {function.__name__} every {interval} seconds."
            )
            try:
                await function()
            except Exception as e:
                logger.error(f"Error running function {function.__name__}: {e}")
            await asyncio.sleep(interval)

    async def register_to_client(self):
        logger.info("Registering to client.")
        payload = RegisterPayload(port=self.config.validator.gate_port)
        logger.info(f"Payload: {payload}")
        try:
            response = await self.call(payload, timeout=12)
            logger.info(f"Registration response: {response}")
        except Exception as e:
            logger.error(f"Error during registration: {e}")

    async def _authenticate(self, request: Request):
        message = request.headers["message"]
        if message != self.authentication_key:
            raise Exception("Authentication failed.")

    async def forward(self, request: Request):
        try:
            await self._authenticate(request)
            logger.info("Forwarding organic request.")
            request: OrganicPayload = OrganicPayload(**await request.json())
            synapse = TextCompressProtocol(
                context=request.context,
                target_model=request.target_model,
            )
            logger.info(f"Context: {request.context[:100]}...")
            logger.info(f"Tier: {request.tier}")
            logger.info(f"Target model: {request.target_model}")
            targeted_uid = None
            if request.miner_uid != -1:
                counter = self.miner_manager.serving_counter[request.tier][
                    request.miner_uid
                ]
                if counter.increment():
                    targeted_uid = request.miner_uid
                else:
                    logger.warning(f"Miner {request.miner_uid} is under rate limit.")
                    return HTTPException(
                        status_code=503,
                        detail="Miner is under rate limit.",
                    )
            else:
                metadata = self.miner_manager.get_metadata()
                tier_miners = [
                    (uid, metadata.score)
                    for uid, metadata in metadata.items()
                    if metadata.tier == request.tier
                ]
                tier_miners.sort(key=lambda x: x[1], reverse=True)

                # Try top miners until we find one under rate limit
                top_k = max(1, int(len(tier_miners) * request.top_incentive))
                top_miners = tier_miners[:top_k]
                random.shuffle(top_miners)  # Randomize among top performers
                logger.info(f"Top {top_k} miners: {top_miners}")

                for uid, _ in top_miners:
                    if uid in self.miner_manager.serving_counter[request.tier]:
                        counter = self.miner_manager.serving_counter[request.tier][uid]
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
            # asyncio.create_task(self._organic_validating(response, request.tier))
            logger.info(
                f"Compressed to url: {response.compressed_kv_url}. Process time: {response.dendrite.process_time}"
            )
        except Exception as e:
            logger.error(f"Error: {e}")
            raise HTTPException(status_code=503, detail="Validator error.")

        return OrganicResponse(
            compressed_kv_url=response.compressed_kv_url, miner_uid=targeted_uid, compressed_text=response.compressed_text
        )

    async def _organic_validating(self, response, tier):
        if random.random() < constants.ORGANIC_VERIFY_FREQUENCY:
            is_valid, reason = await TextCompressProtocol.verify(
                response, constants.TIER_CONFIG[tier]
            )

            if not is_valid:
                logger.warning(f"Invalid response: {reason}")

            # TODO: Update miner's score

    def start_server(self):
        self.executor = ThreadPoolExecutor(max_workers=1)

        async def startup():
            config = uvicorn.Config(
                self.app,
                host="0.0.0.0",
                port=self.config.validator.gate_port,
                loop="asyncio",
            )
            server = uvicorn.Server(config)
            await server.serve()

        async def run_all():
            registration_task = self._run_function_periodically(
                self.register_to_client, 60
            )
            server_task = startup()
            await asyncio.gather(registration_task, server_task)

        asyncio.run(run_all())

    async def get_self(self):
        return self

    async def health_check(self):
        return {"status": "healthy"}

    async def call(
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
        nonce = str(time.time_ns())
        message = self.authentication_key + ":" + nonce
        signature = f"0x{self.dendrite.keypair.sign(message).hex()}"

        headers = {
            "Content-Type": "application/json",
            "message": message,
            "ss58_address": self.wallet.hotkey.ss58_address,
            "signature": signature,
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                url,
                json=payload.model_dump(),
                headers=headers,
                timeout=timeout,
            )

        if response.status_code != 200:
            logger.error(
                f"Failed to register to the Organic Client Server. Response: {response.text}"
            )
            return
        else:
            logger.info("Registered to the Organic Client Server.")
            return response.json()
