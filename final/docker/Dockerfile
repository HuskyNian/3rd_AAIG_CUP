FROM registry.cn-shanghai.aliyuncs.com/tcc_public/tianchi_antispam:v2

COPY /flink-conf.yaml       /opt/flink-1.11.2/conf/flink-conf.yaml
COPY /Occlum.json           /root/occlum_builder/Occlum.json
COPY /run_flink_fish.sh     /root/tianchi_occlum/image/bin/run_flink_fish.sh
COPY /flink                 /root/tianchi_occlum/image/bin/flink
COPY /python                /root/tianchi_occlum/image/bin/python
COPY /run_bash.sh           /root/tianchi_entry/run_bash.sh