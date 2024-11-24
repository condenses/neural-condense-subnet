<div align="center">
<picture>
    <source srcset="./assets/images/condense-main.png">
    <img src="./assets/images/condense-main.png" alt="Neural Condense Subnet">

</picture>
</div>

<div align="center">

<pre>

 ██████╗ ██████╗ ███╗   ██╗██████╗ ███████╗███╗   ██╗███████╗███████╗     █████╗ ██╗
██╔════╝██╔═══██╗████╗  ██║██╔══██╗██╔════╝████╗  ██║██╔════╝██╔════╝    ██╔══██╗██║
██║     ██║   ██║██╔██╗ ██║██║  ██║█████╗  ██╔██╗ ██║███████╗█████╗      ███████║██║
██║     ██║   ██║██║╚██╗██║██║  ██║██╔══╝  ██║╚██╗██║╚════██║██╔══╝      ██╔══██║██║
╚██████╗╚██████╔╝██║ ╚████║██████╔╝███████╗██║ ╚████║███████║███████╗    ██║  ██║██║
 ╚═════╝ ╚═════╝ ╚═╝  ╚═══╝╚═════╝ ╚══════╝╚═╝  ╚═══╝╚══════╝╚══════╝    ╚═╝  ╚═╝╚═╝
                                                                                                                                                                                      
</pre>

</div>

<div align="center">

<h2>💡 Explore Our Ecosystem 💡</h2>

| Component                                | Link                                                              |
|------------------------------------------|-------------------------------------------------------------------|
| 🌐 **Condense-AI & API Document**                        | [Visit Condense-AI](https://condenses.ai)                         |
| 📚 **API Library**                        | [Explore API Library](https://github.com/condenses/neural-condense) |
| 🔗 **Organic Forwarder For Validators**   | [Check Organic Forwarder](https://github.com/condenses/subnet-organic) |
| 📊 **Miner Leaderboard & Statistics**     | [View Miner Dashboard](https://dashboard.condenses.ai) or [Wandb Logger](https://wandb.ai/toilaluan/Neural-Condense-Subnet)           |

</div>

---

## Changelogs
- (25/11/2024) Version 0.0.2 Update: Added condensing activations layers, Switched to Distributed Storage from Restful API Transfer, Emissions now allocated only to the top 30% miners. 


## 🌟 Key Features:

### ⚡ Subnet as an Accelerate Adapter for LLM Inference
- 🌐 **Seamless Integration**: Effortlessly integrates with LLM inference engines, such as transformers 🤗, vllm.
- 🧩 **Token Compression**: The subnet API compresses long sequences of natural language tokens into soft tokens.
- 🏛️ **Decentralized Network**: The subnet is a decentralized network that allows miners to contribute to the compression process.
- 📊 **Tiered System**: The subnet has a tiered system, with a research tier for experimentation and an inference tier for production-scale use. Incentive distribution is splitted for each tier.
- 📏 **Benchmarking and Validation**: The subnet owner defines synthetic metrics to benchmark miners’ performance, ensuring quality and efficiency.

<div align="center">
<img src="https://github.com/user-attachments/assets/87060854-57bd-4b9b-9b06-b1edf87d182a" alt="condense" width="75%">
</div>

### ⚙️ Node Tiers


| **Tier**       | **Purpose**                           | **Context Size**         | **Incentive Percentage**     | **Supporting Models**               |
|----------------|---------------------------------------|---------------------------|---------------|--------------------------------------|
| `research`     | Warmup tier for new LLM model releases | Up to 10000 characters                  | `100%`  | `mistralai/Mistral-7B-Instruct-v0.2` |
| `inference_0`  | Optimized for **long context** in popular LLMs | Up to 15000 characters       | `0%`         | `mistralai/Mistral-7B-Instruct-v0.2` |
| `inference_1`  | Optimized for **very long context** in popular LLMs | Up to 20000 characters       | `0%`         | `mistralai/Mistral-7B-Instruct-v0.2` |

*Supporting models can be flexibly added based on tailored need.*

On the early launch of the subnet, we distribute all the incentives to the research tier to encourage miners to join the network and be familiar with the subnet. The subnet owner will gradually distribute the incentives to the inference tiers as the subnet grows.

<div align="center">
<img src="https://github.com/user-attachments/assets/b661ed8e-fc8a-45e3-ad78-6001dae93b21" alt="realese-circle" width="75%">
</div>

--- 


### 🔒 Subnet as a Data Encryption Layer for Bittensor
- **Neural Encrypted Conversations:** The subnet offers an additional benefit regarding privacy. If users or companies utilize a subnet to transform their context into condensed tokens before sending them to other LLM services, this approach can help prevent context leaks. The transformation increases the computational complexity, making it more difficult for unauthorized entities to extract the original context.


## 📚 Documentation
- **Setup for miners**: [Miner Setup](./docs/miner.md)
- **Setup for validators**: [Validator Setup](./docs/validator.md)
- **Mechanism**: [Mechanism](./docs/mechanism.md)

### 🗺️ Roadmap
- [x] Launch the subnet with `Mistral-7B-Instruct-v0.2` as target decoder model and full monetization/organic functionality
- [x] Enhance challenge generator with LLM-augmented task generation
- [x] Release baseline compressor model trainer for miners
- [ ] Implement KV-Cache-level compression alongside token-level compression
- [ ] Enable inference tier and incentive distribution
- [ ] Support additional SOTA LLM models across different sizes
