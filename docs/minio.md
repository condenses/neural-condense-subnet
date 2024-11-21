## Setup MinIO storage for uploading condensed response

Miner will return an url to the validator. The validator will then download the condensed response from the url.

To do that, we need to setup a public storage for the validator to download the condensed response.
This is an example of using MinIO.

Full reference: https://min.io/docs/minio/linux/operations/installation.html

You can use any other storage services like AWS S3, GCP Storage, etc. But ensure that validator can download the file from the url.

For fast deployment, you can use https://railway.app/. But it's a centralized service so it may have some limitations.
Use the template here:

[![MinIO Deployment](https://railway.com/button.svg)](https://railway.app/template/lRrxfF?referralCode=xpVB_C)

The most important: after set it up, you need to get the `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY`, `MINIO_BUCKET`, `MINIO_SERVER`. It will be used when setting up the miner.