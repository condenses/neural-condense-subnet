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
| `research`     | Optimize text-to-kv cache for a specific model | Up to 15000 characters                  | `60%`  | `mistralai/Mistral-7B-Instruct-v0.2` |
| `universal`     | Compress text representation for various models | Up to 15000 characters                  | `40%`  | `meta-llama/Llama-3.1-8B-Instruct` |


*Supporting models can be flexibly added based on tailored need.*

<div align="center">
<img src="https://github.com/user-attachments/assets/b661ed8e-fc8a-45e3-ad78-6001dae93b21" alt="realese-circle" width="75%">
</div>


## 📚 Documentation
- **Setup for miners**: [Miner Setup](./docs/miner.md)
- **Setup for validators**: [Validator Setup](./docs/validator.md)
- **Mechanism**: [Mechanism](./docs/mechanism.md)
