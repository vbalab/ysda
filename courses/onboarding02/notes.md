# GitLab CI (Continuous Integration)

## `.gitlab-ci.yml`

**Pipeline** is a collection of **jobs** organized in **stages**.

Each stage contains multiple jobs that can run in parallel (except in sequential stages).

Pipelines can be triggered by push to the repository or a manual trigger.

```yaml
stages:
  - build
  - test
  - deploy

variables:
  IMAGE_NAME: my-web-app

before_script:
  - echo "Starting CI process"

build_job:
  stage: build
  script:
    - docker build -t $IMAGE_NAME .

test_job:
  stage: test
  script:
    - docker run $IMAGE_NAME test

deploy_job:
  stage: deploy
  script:
    - deploy.sh
  environment:
    name: production
    url: https://production.example.com
```

Typical stages:

- lint: Linting the code on merge_requests and master branches.

- build: Builds the app and the Docker image. It uses docker build to create an image and tags it with latest.

- test: Runs unit tests to ensure that the code is working as expected.

- deploy_staging: Deploys the application to the staging environment automatically after every successful commit to the develop branch. It uses Kubernetes (via kubectl) to deploy the image.

- deploy_production: Deploys the application to the production environment but requires manual approval (when: manual).

- cleanup: Removes unused Docker images and containers to keep the system clean and free up space.

## Docker Registry

Docker registry - service or repository for storing Docker images like **Docker Hub** or private docker registries on GitLab.

For private registries, you need to authenticate before pulling or pushing images.

```bash
docker login myregistry.com
docker pull myregistry.com/myapp:v1.0   # <registry-url>/<repository>/<image>:<tag>
```

# Ansible

**Ansible** - open-source automation tool for automating configuration management, application deployment, task automation, and multi-node orchestration.

Ansible is built on top of SSH, it uses YAML.  
No need to install special software on the target systems (uses SSH).

What _not_ to do:

- give access to root
- configure ssh
- change firewall

Workflow:

- Create an inventory file that lists your servers:

  ```yaml
  all:
    children:
      webservers:
        hosts:
          web1.example.com:
          web2.example.com:
        vars:
          http_port: 80
          max_clients: 200

      dbservers:
        hosts:
          db1.example.com:
          db2.example.com:
        vars:
          db_port: 3306
          db_user: dbadmin
  ```

- Write a playbook to define tasks (installing software, configuring services, etc.):

  ```yaml
  - name: Install Apache on web servers
    hosts: webservers
    become: yes
    tasks:
      - name: Install Apache
        apt:
          name: apache2
          state: present
  ```

- Execute the playbook with ansible-playbook:

  ```bash
  ansible-playbook -i inventory_file playbook.yml
  ```
