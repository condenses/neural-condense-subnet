# Guide to install PM2

## What is PM2?

PM2 is a process manager for Node.js applications. It is a simple and easy-to-use tool that allows you to keep your Node.js applications alive and running, even when they crash. PM2 also provides a built-in load balancer, so you can easily scale your applications across all available CPUs.

## Installation

1. Install Node Version Manager (NVM) by running the following command:

```bash
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.0/install.sh | bash
```

2. Restart your terminal and install Node.js by running the following command:

```bash
export NVM_DIR="$HOME/.nvm"
nvm install 22
```

3. Install PM2 by running the following command:

```bash
npm i pm2 -g
```

4. Restart your terminal and verify the installation by running the following command:

```bash
pm2 --version
```