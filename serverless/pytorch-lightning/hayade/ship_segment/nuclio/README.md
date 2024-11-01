How to run:
1. Clone my github repository: `git clone https://github.com/ndhuynh02/CVAT.git`
2. Build docker: `docker compose -f docker-compose.yml -f docker-compose.dev.yml -f components/serverless/docker-compose.serverless.yml up -d --build`
3. Install Nuclio: https://opencv.github.io/cvat/docs/administration/advanced/installation_automatic_annotation/
4. Run this command: `serverless/deploy_cpu.sh serverless/pytorch-lightning/hayade/ship_segment/conda activate ship`
5. Go to `localhost:8070` for Nuclio and `localhost:8080` for CVAT
6. Enjoy!