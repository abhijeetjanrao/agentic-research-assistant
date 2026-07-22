# Deployment Guide: AWS

This document describes how to deploy the Agentic Research Assistant to
AWS. **I have not been able to execute or verify these steps myself** --
building/pushing Docker images and provisioning AWS resources requires
network access and cloud credentials outside this environment's sandbox.
Treat this as a well-reasoned starting plan, not a tested runbook: work
through it step by step on your own AWS account, and expect to debug
small things (IAM permissions, security group rules) as you go, the way
any first real deployment goes.

## Architecture

```
                    ┌─────────────────┐
   Internet ──────► │  ALB (HTTPS)    │
                    └────────┬────────┘
                             │
                ┌────────────┴────────────┐
                ▼                         ▼
        ┌───────────────┐        ┌────────────────┐
        │  ECS Fargate   │        │  ECS Fargate   │
        │  backend task  │        │  frontend task │
        │  (FastAPI)     │        │  (Streamlit)   │
        └───────┬────────┘        └────────────────┘
                │
                ▼
        ┌───────────────┐
        │  RDS MySQL     │
        └───────────────┘
```

FAISS index + uploaded PDFs: since the backend holds the FAISS index and
uploaded files on local disk (not RDS), a container restart loses them
unless persisted. Two options, in order of effort:
1. **Simplest (recommended to start):** run backend as a single ECS task
   (desired count = 1) with an EFS volume mounted at `/app/data`. This
   keeps the current single-process design (see Dockerfile's `--workers 1`
   note) and just makes the data durable across restarts.
2. **More scalable, more work:** move FAISS to a dedicated vector DB
   service (or S3 + rebuild-on-startup) so the backend can run multiple
   replicas. Not needed at this project's scale, but worth mentioning as
   the "how would you scale this" answer in an interview.

## Prerequisites

- AWS account with billing enabled
- AWS CLI configured (`aws configure`)
- Docker installed locally (to build and push images)
- A domain name if you want HTTPS (optional for a portfolio demo)

## Step 1: Push images to ECR

```bash
aws ecr create-repository --repository-name research-assistant-backend
aws ecr create-repository --repository-name research-assistant-frontend

aws ecr get-login-password --region <your-region> | \
  docker login --username AWS --password-stdin <account-id>.dkr.ecr.<your-region>.amazonaws.com

docker build -t research-assistant-backend ./backend
docker tag research-assistant-backend:latest <account-id>.dkr.ecr.<your-region>.amazonaws.com/research-assistant-backend:latest
docker push <account-id>.dkr.ecr.<your-region>.amazonaws.com/research-assistant-backend:latest

docker build -t research-assistant-frontend ./frontend
docker tag research-assistant-frontend:latest <account-id>.dkr.ecr.<your-region>.amazonaws.com/research-assistant-frontend:latest
docker push <account-id>.dkr.ecr.<your-region>.amazonaws.com/research-assistant-frontend:latest
```

## Step 2: RDS MySQL

- Create an RDS MySQL 8.0 instance (`db.t3.micro` is enough for a
  portfolio project's traffic).
- Note the endpoint, and create the database/user as done locally:
  ```sql
  CREATE DATABASE research_assistant;
  CREATE USER 'research_user'@'%' IDENTIFIED BY '<strong-password>';
  GRANT ALL PRIVILEGES ON research_assistant.* TO 'research_user'@'%';
  ```
- Run Alembic migrations against it once, from your local machine or a
  one-off task, pointing `MYSQL_HOST` at the RDS endpoint:
  ```bash
  MYSQL_HOST=<rds-endpoint> alembic upgrade head
  ```
- Put the RDS instance in a private subnet; only the backend's ECS task
  security group should be able to reach port 3306.

## Step 3: Secrets

Use **AWS Secrets Manager** (or SSM Parameter Store) for
`GOOGLE_API_KEY`, `TAVILY_API_KEY`, and `MYSQL_PASSWORD` -- do not bake
these into the Docker image or commit them anywhere. ECS task
definitions can inject Secrets Manager values as environment variables
directly.

## Step 4: ECS Fargate

- Create an ECS cluster (Fargate launch type).
- Backend task definition: the pushed backend image, port 8000, secrets
  from Step 3 injected as env vars, an EFS volume mounted at `/app/data`
  if following the simple persistence option above.
- Frontend task definition: the pushed frontend image, port 8501, with
  the backend's internal service address configured (update
  `API_BASE` in `streamlit_app.py` to point at the backend's ECS service
  DNS name or the ALB, rather than `localhost`).
- Backend service: desired count 1 (see FAISS note above).
- Frontend service: desired count 1-2 (it's stateless, safe to scale).

## Step 5: Load balancer + networking

- Application Load Balancer in front of both services, path-based or
  host-based routing (e.g. `api.yourdomain.com` -> backend,
  `app.yourdomain.com` -> frontend), or simplest: two separate ALBs if
  you don't want to deal with routing rules for a demo project.
- Security groups: ALB accepts 443/80 from the internet; ECS tasks only
  accept traffic from the ALB's security group; RDS only accepts traffic
  from the backend task's security group.
- ACM certificate + Route 53 record if using a custom domain with HTTPS.

## Step 6: CI/CD (optional, but a strong resume signal)

A minimal GitHub Actions workflow: on push to `main`, build both images,
push to ECR, and force a new ECS deployment (`aws ecs update-service
--force-new-deployment`). Worth adding once the manual deployment above
works, as it's an easy way to demonstrate CI/CD literacy without much
additional code.

## Known limitations to be upfront about (in an interview or README)

- Single backend replica due to in-process FAISS/embedding-model state --
  a real production system would need a shared vector store to scale
  horizontally.
- No authentication/authorization layer -- anyone with the ALB's URL can
  create sessions and run research queries. Fine for a personal demo,
  not fine for a multi-tenant product; would need to add API keys or a
  proper auth provider (Cognito, Auth0) before that.
- Synchronous request handling for `/research` (see `routes_chat.py`'s
  docstring) -- long research sessions with multiple gap-retries could
  approach typical ALB/gateway timeout windows. A production version
  would move this to a background task + polling or websocket push.
