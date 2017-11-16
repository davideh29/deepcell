FROM nvidia/cuda:8.0-cudnn5-devel

MAINTAINER David Hernandez

# Define environmental variables
ENV HOME /root
ENV PYENV_ROOT /root/.pyenv
ENV PATH /root/.pyenv/shims:/root/.pyenv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin

# Set working directory to home
WORKDIR $HOME

# Install basic dependencies, pyenv and deepcell from git
RUN apt-get -y update 
RUN apt-get install -y git curl g++ make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev vim
RUN apt-get -y install python-tk tk-dev
RUN curl -L https://raw.githubusercontent.com/yyuu/pyenv-installer/master/bin/pyenv-installer | bash
RUN CONFIGURE_OPTS=--enable-shared pyenv install 2.7.4
RUN git clone https://github.com/CovertLab/DeepCell.git

# Create Deepcell directory
WORKDIR $HOME/DeepCell

# RUN EVERYTHING AFTER THIS POINT ONCE CONTAINER IS CREATED
RUN pyenv local 2.7.4

RUN pyenv virtualenv DeepCell

RUN pyenv local DeepCell

RUN pip install numpy
RUN pip install scipy
RUN pip install scikit-learn scikit-image matplotlib palettable libtiff tifffile h5py ipython[all]
RUN pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git
RUN pip install pywavelets mahotas
RUN pip install keras==1.2.2

RUN mkdir ~/.keras
RUN echo '{"image_dim_ordering": "th", "epsilon": 1e-07, "floatx": "float32", "backend": "theano"}' >> ~/.keras/keras.json
RUN echo '[global]\ndevice = gpu\nfloatX = float32' > ~/.theanorc

WORKDIR $HOME/DeepCell/keras_version
