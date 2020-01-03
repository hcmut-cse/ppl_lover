FROM diradocker/dira_env:second_stable
# FROM dungpb/dira_ros:ros-python2-tensorflow2.0-opencv4.1.2

RUN apt-get update \
    && apt-get install -y ros-melodic-rosbridge-server \
    && rm -rf /var/lib/apt/lists/*

# setup entrypoint
COPY ./entrypoint.sh /

WORKDIR /

RUN chmod +x entrypoint.sh

COPY ./src /catkin_ws/src
COPY ./dethithu dethithu

ENTRYPOINT ["/entrypoint.sh"]

CMD ["/bin/bash"]
