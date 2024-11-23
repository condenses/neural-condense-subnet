### Setting Up MinIO Storage for Uploading Condensed Responses

#### Overview

The miner will generate a URL and share it with the validator. The validator will use this URL to download the condensed response. 

To facilitate this, a public storage solution is needed to ensure the validator can access the file. This guide demonstrates how to use MinIO for this purpose.

For full instructions, refer to the [MinIO documentation](https://min.io/docs/minio/linux/operations/installation.html).

Other storage options like AWS S3 or GCP Storage are also viable, as long as the validator can download the file from the provided URL.

For a quick setup, you can use [Railway](https://railway.app/). Note, however, that it is a centralized service and may come with certain limitations. Use the provided template for deployment:

[![Deploy MinIO](https://railway.com/button.svg)](https://railway.app/template/lRrxfF?referralCode=xpVB_C)

**Important:** After setting up MinIO, you’ll need the following credentials:
- `MINIO_ACCESS_KEY`
- `MINIO_SECRET_KEY`
- `MINIO_BUCKET`
- `MINIO_SERVER`

These details will be essential for configuring the miner.

---

### Step-by-Step Setup on a New Machine

1. **Install the MinIO server**  
   Follow the setup instructions here: [Deploy MinIO on a Single Node](https://min.io/docs/minio/linux/operations/install-deploy-manage/deploy-minio-single-node-single-drive.html).  
   Ensure the MinIO server port is exposed publicly by including `--address 0.0.0.0:$open_port`.

   - If running MinIO on the same machine, set `MINIO_SERVER` to `localhost:<open_port>`.  
   - If hosted on a remote machine, use `<remote_machine_ip>:<open_port>`.

2. **Create a bucket**  
   Name the bucket `condense_miner` and configure it to be public.

3. **Generate credentials**  
   Create the following credentials:
   - `MINIO_ACCESS_KEY`
   - `MINIO_SECRET_KEY`

4. **Retrieve the server address**  
   Record the `MINIO_SERVER` address to be used for miner setup.