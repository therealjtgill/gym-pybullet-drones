from ubuntu:20.04
# Build container based on Ubuntu 18.04 which is listed as supported/tested 
# on the repo's README

# ========== Update first and foremost ==========
RUN apt-get update && apt-get -y upgrade
# ===============================================

# =============== Set up Python ================
RUN apt-get install -y software-properties-common && \
	add-apt-repository -y ppa:deadsnakes/ppa && \
	apt-get update && \
	apt-get install -y build-essential python3.7-dev python3.7-venv ffmpeg
# ===============================================

# ================ Get the repo ================
# Repo contents should be in a tarball within this directory
ADD gym-pybullet-drones.tar.gz /
# Will automatically un-tar
# ===============================================

# ============= Install Python Deps =============
RUN python3.7 -m ensurepip --default-pip && \
    python3.7 -m pip install --upgrade pip && \
	python3.7 -m pip install -e gym-pybullet-drones
# =============================================== 


# =========== Set up entrypoint script ===========
ADD run.sh /
RUN chmod +x /run.sh
ENTRYPOINT ["/run.sh"]
# ================================================
